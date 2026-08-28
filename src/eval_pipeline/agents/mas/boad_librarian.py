"""BOAD MAS variant — fixed plan + code_navigator replaced with Repo Librarian.

Two manipulations on :mod:`mas.boad` (the paper's k=2 baseline):
  1. No plan-generation LLM call — the orchestrator runs a fixed 6-step plan
     (:data:`mas.prompts._variants._FIXED_PLAN`).
  2. ``code_navigator`` → persistent Repo Librarian (view-only, "locator not
     analyst", :class:`SharedSubAgent` memory). ``issue_analyzer`` untouched.

Layout (mirrors :mod:`mas.boad`):
    I.   CONFIG       — :class:`BOADLibrarianConfig`.
    II.  SUB-AGENTS   — :data:`SUBAGENTS`; orchestrator + issue_analyzer
                        prompts from :mod:`mas.prompts._boad`, librarian from
                        :mod:`mas.prompts._librarian`, fixed plan from
                        :mod:`mas.prompts._variants`.
    III. TOOLS        — issue_analyzer: bash + editor; librarian: view-only +
                        view side-channel.
    IV.  HARNESS      — :class:`BOADLibrarianMASRunner`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agents import function_tool

from eval_pipeline.agents.base import SubAgentSpec
from eval_pipeline.agents.bases.delegation import invoke_agent_logged
from eval_pipeline.agents.history import DEFAULT_BRANCH_THRESHOLD_CHARS, maybe_wrap_last_n
from eval_pipeline.agents.mas._base import MASRunnerBase
from eval_pipeline.agents.mas._delegation import make_ephemeral_delegate
from eval_pipeline.agents.mas._librarian_role import (
    LibrarianRoleMixin,
    make_ephemeral_librarian_call,
    make_librarian_call,
    make_view_command_submit_tool,
)
from eval_pipeline.agents.mas._runtime import init_mas_runtime
from eval_pipeline.agents.mas._tool_wrap import _wrap_tools_for_boad_template
from eval_pipeline.agents.mas.prompts._boad import (
    _ISSUE_ANALYZER_DOCSTRING,
    _ISSUE_ANALYZER_INSTANCE,
    _ORCHESTRATOR_INSTANCE,
    _ORCHESTRATOR_SYSTEM,
    _SUBAGENT_SYSTEM_TEMPLATE,
)
from eval_pipeline.agents.mas.prompts._librarian import (
    _LIBRARIAN_SYSTEM,
    _LIBRARIAN_SYSTEM_NOPERSIST,
    _LIBRARIAN_SYSTEM_NOPOINTER,
    _LIBRARIAN_SYSTEM_NOSCOPE_BOAD,
)
from eval_pipeline.agents.mas.prompts._variants import (
    _FIXED_PLAN,
    _FIXED_PLAN_NOPERSIST,
    _FIXED_PLAN_NOSCOPE,
    _LIBRARIAN_DOCSTRING,
    _LIBRARIAN_DOCSTRING_NOPERSIST,
    _LIBRARIAN_DOCSTRING_NOSCOPE,
    LIBRARIAN_TOOL_DOC,
    LIBRARIAN_TOOL_DOC_NOPERSIST,
    LIBRARIAN_TOOL_DOC_NOSCOPE,
)
from eval_pipeline.agents.openhands import (
    NudgePolicy,
    make_view_only_bash,
    make_view_only_str_replace_editor,
)
from eval_pipeline.agents.runtime import CreateMode

logger = logging.getLogger(__name__)


# ── I. CONFIG ────────────────────────────────────────────────────────────
@dataclass
class BOADLibrarianConfig:
    model: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    run_timeout: int = 3600
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "dummy"
    max_iterations: int = 1000
    max_observation_length: int = 30000
    max_nudges: int = 2
    bash_timeout: int = 1800
    seed: int | None = None
    branch_threshold_chars: int = DEFAULT_BRANCH_THRESHOLD_CHARS
    drop_below_threshold: bool = True
    caveman_level: str | None = None
    last_n_observations: int | dict[str, int] | None = None


# ── II. SUB-AGENTS ───────────────────────────────────────────────────────
def _subagents_overview(librarian_docstring: str = _LIBRARIAN_DOCSTRING) -> str:
    """Overview block for the orchestrator prompt; parameterised so the
    librarian ablations can swap the docstring without touching the default
    output text."""
    return (
        f"- name: issue_analyzer -- {_ISSUE_ANALYZER_DOCSTRING}\n"
        f"- name: librarian -- {librarian_docstring}"
    )


_SUBAGENTS_OVERVIEW = _subagents_overview()

SUBAGENTS: dict[str, SubAgentSpec] = {
    "orchestrator": SubAgentSpec(
        name="orchestrator", system_prompt=_ORCHESTRATOR_SYSTEM,
        invoke_template=_ORCHESTRATOR_INSTANCE, stop_at=["submit"], base_tools=None,
    ),
    "issue_analyzer": SubAgentSpec(
        name="issue_analyzer", system_prompt=_SUBAGENT_SYSTEM_TEMPLATE,
        invoke_template=_ISSUE_ANALYZER_INSTANCE, stop_at=["submit_subagent"],
        base_tools=["bash", "str_replace_editor"],
    ),
    "librarian": SubAgentSpec(
        name="librarian", system_prompt=_LIBRARIAN_SYSTEM,
        invoke_template=None, stop_at=["submit_result"], base_tools=[],
    ),
}


# ── IV. HARNESS ──────────────────────────────────────────────────────────
class BOADLibrarianMASRunner(LibrarianRoleMixin, MASRunnerBase):
    CONFIG = BOADLibrarianConfig
    LABEL = "BOADLibrarianMAS"
    CAVEMAN_ROLES = ("orchestrator", "issue_analyzer", "librarian")
    LIBRARIAN_ID = "librarian"
    SUBAGENTS = SUBAGENTS

    async def run_with_executor(self, instruction, work_dir, *, executor, runtime):
        assert self.docker is not None, "Call setup() first"
        mr = await init_mas_runtime(self, label=self.label)
        _cav = self.caveman_fn()
        self.note_last_n()

        # Leave-one-out ablation selection. ``None`` picks the exact default
        # strings, so the non-ablated path is byte-identical to L2.
        ablation = self.librarian_ablation
        lib_system = {
            "persistence": _LIBRARIAN_SYSTEM_NOPERSIST,
            "pointer": _LIBRARIAN_SYSTEM_NOPOINTER,
            "scoped": _LIBRARIAN_SYSTEM_NOSCOPE_BOAD,
        }.get(ablation, _LIBRARIAN_SYSTEM)
        lib_doc = {
            "persistence": LIBRARIAN_TOOL_DOC_NOPERSIST,
            "scoped": LIBRARIAN_TOOL_DOC_NOSCOPE,
        }.get(ablation, LIBRARIAN_TOOL_DOC)
        lib_docstring = {
            "persistence": _LIBRARIAN_DOCSTRING_NOPERSIST,
            "scoped": _LIBRARIAN_DOCSTRING_NOSCOPE,
        }.get(ablation, _LIBRARIAN_DOCSTRING)
        plan = {
            "persistence": _FIXED_PLAN_NOPERSIST,
            "scoped": _FIXED_PLAN_NOSCOPE,
        }.get(ablation, _FIXED_PLAN)
        print(f"[{self.label}] Using fixed plan ({len(plan)} chars)")

        @function_tool
        async def submit_subagent(result: str) -> str:
            return result

        pending_view_output: list[str] = []
        if ablation == "pointer":
            # No view side-channel: 1-param submit, code pasted inline in
            # ``result``. The buffer stays empty so no prepend ever fires.
            @function_tool(name_override="submit_result")
            async def lib_submit(result: str) -> str:
                return result
        else:
            lib_submit = make_view_command_submit_tool(
                self.docker, pending_view_output,
            )

        # issue_analyzer (edit-capable ephemeral sub)
        ia_spec = SUBAGENTS["issue_analyzer"]
        ia_llm = maybe_wrap_last_n(mr.llm_model, self.last_n_observations, "issue_analyzer")
        ia_agent, ia_tools = self.base_agent_builder(
            docker=self.docker, work_dir=work_dir, llm_model=ia_llm,
            agent_name=ia_spec.name,
            system_prompt=_cav(ia_spec.system_prompt.format(
                role="issue analysis",
                main_task="analyze issue descriptions and extract key information",
                problem_statement=instruction,
            )),
            extra_tools=[submit_subagent], stop_at_tool_names=ia_spec.stop_at,
            max_observation_length=runtime.max_observation_length,
            bash_timeout=runtime.bash_timeout, base_tools=ia_spec.base_tools,
        )
        _wrap_tools_for_boad_template(ia_tools, fn_call_rules=False)
        ia_delegate = make_ephemeral_delegate(
            mr=mr, executor=executor, agent=ia_agent, agent_role="issue_analyzer",
            max_nudges=runtime.max_nudges,
            format_input=lambda c: _ISSUE_ANALYZER_INSTANCE.format(context=c),
            text_only_policy=NudgePolicy(max_retries=2),
        )

        # librarian (persistent by default; ephemeral under "persistence")
        lib_spec = SUBAGENTS["librarian"]
        lib_llm = maybe_wrap_last_n(mr.llm_model, self.last_n_observations, "librarian")
        lib_bash = make_view_only_bash(
            self.docker, work_dir,
            max_observation_length=runtime.max_observation_length,
            timeout=runtime.bash_timeout,
        )
        lib_sre = make_view_only_str_replace_editor(
            self.docker, work_dir,
            max_observation_length=runtime.max_observation_length,
        )
        librarian_agent, lib_tools = self.base_agent_builder(
            docker=self.docker, work_dir=work_dir, llm_model=lib_llm,
            agent_name=lib_spec.name, system_prompt=_cav(lib_system),
            extra_tools=[lib_bash, lib_sre, lib_submit],
            stop_at_tool_names=lib_spec.stop_at,
            max_observation_length=runtime.max_observation_length,
            bash_timeout=runtime.bash_timeout, base_tools=[],
        )
        _wrap_tools_for_boad_template(lib_tools, fn_call_rules=False)
        if ablation == "persistence":
            shared_librarian = None
            _lib_call = make_ephemeral_librarian_call(
                mr=mr, executor=executor, librarian_agent=librarian_agent,
                librarian_id=self.LIBRARIAN_ID, max_turns=self.max_iterations,
                pending_view_output=pending_view_output,
                max_nudges=runtime.max_nudges,
                text_only_policy=NudgePolicy(max_retries=2),
            )
        else:
            shared_librarian = self._make_navigator(
                navigator_agent=librarian_agent, work_dir=work_dir,
            )
            _lib_call = make_librarian_call(
                mr=mr, shared_librarian=shared_librarian, librarian_id=self.LIBRARIAN_ID,
                caller_role="orchestrator", max_turns=self.max_iterations,
                pending_view_output=pending_view_output, max_nudges=runtime.max_nudges,
                text_only_policy=NudgePolicy(max_retries=2),
            )

        @function_tool
        async def issue_analyzer(context: str) -> str:
            return await ia_delegate(context)

        async def _librarian(query: str) -> str:
            return await _lib_call(query)

        _librarian.__doc__ = lib_doc
        librarian = function_tool(name_override="librarian")(_librarian)

        # orchestrator (top-level)
        orch_spec = SUBAGENTS["orchestrator"]
        orch_llm = maybe_wrap_last_n(mr.llm_model, self.last_n_observations, "orchestrator")
        orchestrator, orch_tools = self.base_agent_builder(
            docker=self.docker, work_dir=work_dir, llm_model=orch_llm,
            agent_name=orch_spec.name,
            system_prompt=_cav(orch_spec.system_prompt.format(
                subagents_overview=_subagents_overview(lib_docstring),
            )),
            extra_tools=[issue_analyzer, librarian],
            stop_at_tool_names=orch_spec.stop_at,
            max_observation_length=runtime.max_observation_length,
            bash_timeout=runtime.bash_timeout,
        )
        _wrap_tools_for_boad_template(orch_tools, fn_call_rules=True)

        first_user_msg = orch_spec.invoke_template.format(
            work_dir=work_dir, problem_statement=instruction, plan=plan,
        )

        async def _execute_orch(inst, hooks):
            return await executor.execute_one(
                agent=orchestrator, input=first_user_msg, hooks=hooks,
                max_turns=self.max_iterations,
                text_only_policy=NudgePolicy(max_retries=2),
            )

        print(f"[{self.label}] Running orchestrator...")
        try:
            await invoke_agent_logged(**mr.logged(
                agent_type="orchestrator",
                agent_obj_factory=lambda: orchestrator, execute=_execute_orch,
                agent_id="orchestrator_0", mode=CreateMode.CREATE,
                max_nudges=runtime.max_nudges,
            ))
        finally:
            # Ephemeral librarian ("persistence" ablation) has nothing to close.
            cleanup = getattr(shared_librarian, "close", None) if shared_librarian else None
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as e:  # noqa: BLE001
                    logger.warning("navigator close failed: %s", e)
