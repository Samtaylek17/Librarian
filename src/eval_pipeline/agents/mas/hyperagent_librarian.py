"""HyperAgent MAS variant — Navigator replaced with a persistent Repo Librarian.

Identical to :mod:`mas.hyperagent` except the **Navigator** sub-agent is
swapped for the persistent **Repo Librarian** (locator-not-analyst,
:class:`SharedSubAgent` memory). Editor and Executor are untouched. In the
planner's prompt only the names change (Codebase Navigator → Repo Librarian,
delegate_to_navigator → delegate_to_librarian); the workflow is the baseline.

Layout (mirrors :mod:`mas.hyperagent`):
    I.   CONFIG       — :class:`HyperAgentLibrarianConfig`.
    II.  SUB-AGENTS   — :data:`SUBAGENTS`; planner prompt from
                        :mod:`mas.prompts._variants`, editor/executor from
                        :mod:`mas.prompts._hyperagent`, librarian from
                        :mod:`mas.prompts._librarian`.
    III. TOOLS        — librarian: view-only + view side-channel; editor:
                        edit; executor: bash-only.
    IV.  HARNESS      — :class:`HyperAgentLibrarianMASRunner`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agents import Agent, function_tool

from eval_pipeline.agents.base import SubAgentSpec
from eval_pipeline.agents.bases.delegation import invoke_agent_logged
from eval_pipeline.agents.history import DEFAULT_BRANCH_THRESHOLD_CHARS, maybe_wrap_last_n
from eval_pipeline.agents.llm.profiles import render_tool_examples
from eval_pipeline.agents.mas._base import MASRunnerBase
from eval_pipeline.agents.mas._delegation import make_ephemeral_delegate
from eval_pipeline.agents.mas._librarian_role import (
    LibrarianRoleMixin,
    make_ephemeral_librarian_call,
    make_librarian_call,
    make_view_command_submit_tool,
)
from eval_pipeline.agents.mas._runtime import init_mas_runtime
from eval_pipeline.agents.mas.prompts._hyperagent import _EDITOR_SYSTEM, _EXECUTOR_SYSTEM
from eval_pipeline.agents.mas.prompts._librarian import (
    _LIBRARIAN_SYSTEM,
    _LIBRARIAN_SYSTEM_NOPERSIST,
    _LIBRARIAN_SYSTEM_NOPOINTER,
    _LIBRARIAN_SYSTEM_NOSCOPE,
)
from eval_pipeline.agents.mas.prompts._variants import (
    _PLANNER_INSTANCE_TEMPLATE,
    _PLANNER_SYSTEM,
    _PLANNER_SYSTEM_NOPERSIST,
    _PLANNER_SYSTEM_NOSCOPE,
    DELEGATE_TO_LIBRARIAN_DOC,
    DELEGATE_TO_LIBRARIAN_DOC_NOPERSIST,
    DELEGATE_TO_LIBRARIAN_DOC_NOSCOPE,
)
from eval_pipeline.agents.runtime import CreateMode

logger = logging.getLogger(__name__)


# ── I. CONFIG ────────────────────────────────────────────────────────────
@dataclass
class HyperAgentLibrarianConfig:
    model: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    run_timeout: int = 3600
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "dummy"
    planner_max_turns: int = 1000
    subagent_max_turns: int = 1000
    max_observation_length: int = 16000
    max_nudges: int = 2
    bash_timeout: int = 1800
    seed: int | None = None
    branch_threshold_chars: int = DEFAULT_BRANCH_THRESHOLD_CHARS
    drop_below_threshold: bool = True
    caveman_level: str | None = None
    last_n_observations: int | dict[str, int] | None = None


# ── II. SUB-AGENTS ───────────────────────────────────────────────────────
SUBAGENTS: dict[str, SubAgentSpec] = {
    "planner": SubAgentSpec(
        name="Planner", system_prompt=_PLANNER_SYSTEM,
        invoke_template=_PLANNER_INSTANCE_TEMPLATE, stop_at=["finish"], base_tools=[],
    ),
    "librarian": SubAgentSpec(
        name="Librarian", system_prompt=_LIBRARIAN_SYSTEM,
        invoke_template=None, stop_at=["submit_result"], base_tools=[],
    ),
    "editor": SubAgentSpec(
        name="Editor", system_prompt=_EDITOR_SYSTEM,
        invoke_template=None, stop_at=["submit_result"], base_tools=[],
    ),
    "executor": SubAgentSpec(
        name="Executor", system_prompt=_EXECUTOR_SYSTEM,
        invoke_template=None, stop_at=["submit_result"], base_tools=[],
    ),
}


# ── IV. HARNESS ──────────────────────────────────────────────────────────
class HyperAgentLibrarianMASRunner(LibrarianRoleMixin, MASRunnerBase):
    CONFIG = HyperAgentLibrarianConfig
    LABEL = "HyperAgentLibrarianMAS"
    CAVEMAN_ROLES = ("planner", "librarian", "editor", "executor")
    LIBRARIAN_ID = "librarian"
    SUBAGENTS = SUBAGENTS

    async def run_with_executor(self, instruction, work_dir, *, executor, runtime):
        assert self.docker is not None, "Call setup() first"
        mr = await init_mas_runtime(self, label=self.label)
        _cav = self.caveman_fn()
        self.note_last_n()

        def _fmt(prompt: str) -> str:
            return _cav(render_tool_examples(prompt, self.model))

        # Leave-one-out ablation selection. ``None`` picks the exact default
        # strings, so the non-ablated path is byte-identical to L2.
        ablation = self.librarian_ablation
        lib_system = {
            "persistence": _LIBRARIAN_SYSTEM_NOPERSIST,
            "pointer": _LIBRARIAN_SYSTEM_NOPOINTER,
            "scoped": _LIBRARIAN_SYSTEM_NOSCOPE,
        }.get(ablation, _LIBRARIAN_SYSTEM)
        lib_doc = {
            "persistence": DELEGATE_TO_LIBRARIAN_DOC_NOPERSIST,
            "scoped": DELEGATE_TO_LIBRARIAN_DOC_NOSCOPE,
        }.get(ablation, DELEGATE_TO_LIBRARIAN_DOC)
        planner_system = {
            "persistence": _PLANNER_SYSTEM_NOPERSIST,
            "scoped": _PLANNER_SYSTEM_NOSCOPE,
        }.get(ablation, _PLANNER_SYSTEM)

        @function_tool
        async def submit_result(result: str) -> str:
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

        from eval_pipeline.agents.tools import (
            make_tools as _make_tools,
            make_view_only_bash,
            make_view_only_str_replace_editor,
        )
        nav_bash = make_view_only_bash(
            self.docker, work_dir,
            max_observation_length=runtime.max_observation_length,
            timeout=runtime.bash_timeout,
        )
        nav_sre = make_view_only_str_replace_editor(
            self.docker, work_dir,
            max_observation_length=runtime.max_observation_length,
        )
        full = list(_make_tools(
            self.docker, work_dir,
            max_observation_length=runtime.max_observation_length,
            bash_timeout=runtime.bash_timeout,
        ))
        ed_sre = next(t for t in full if getattr(t, "name", None) == "str_replace_editor")
        ex_bash = next(t for t in full if getattr(t, "name", None) == "bash")

        # librarian (persistent by default; ephemeral under "persistence")
        lib_spec = SUBAGENTS["librarian"]
        lib_llm = maybe_wrap_last_n(mr.llm_model, self.last_n_observations, "librarian")
        librarian_agent, _ = self.base_agent_builder(
            docker=self.docker, work_dir=work_dir, llm_model=lib_llm,
            agent_name=lib_spec.name, system_prompt=_fmt(lib_system),
            extra_tools=[nav_bash, nav_sre, lib_submit],
            stop_at_tool_names=lib_spec.stop_at,
            max_observation_length=runtime.max_observation_length,
            bash_timeout=runtime.bash_timeout, base_tools=[],
        )
        if ablation == "persistence":
            shared_librarian = None
            _lib_call = make_ephemeral_librarian_call(
                mr=mr, executor=executor, librarian_agent=librarian_agent,
                librarian_id=self.LIBRARIAN_ID,
                max_turns=self.subagent_max_turns,
                pending_view_output=pending_view_output,
                max_nudges=runtime.max_nudges,
            )
        else:
            shared_librarian = self._make_navigator(
                navigator_agent=librarian_agent, work_dir=work_dir,
            )
            _lib_call = make_librarian_call(
                mr=mr, shared_librarian=shared_librarian, librarian_id=self.LIBRARIAN_ID,
                caller_role="planner", max_turns=self.subagent_max_turns,
                pending_view_output=pending_view_output, max_nudges=runtime.max_nudges,
            )

        # editor / executor (ephemeral)
        role_tools = {
            "editor": [nav_bash, ed_sre, submit_result],
            "executor": [ex_bash, submit_result],
        }
        delegates: dict[str, Any] = {}
        for role in ("editor", "executor"):
            spec = SUBAGENTS[role]
            role_llm = maybe_wrap_last_n(mr.llm_model, self.last_n_observations, role)
            agent = self.base_agent_builder(
                docker=self.docker, work_dir=work_dir, llm_model=role_llm,
                agent_name=spec.name, system_prompt=_fmt(spec.system_prompt),
                extra_tools=role_tools[role], stop_at_tool_names=spec.stop_at,
                max_observation_length=runtime.max_observation_length,
                bash_timeout=runtime.bash_timeout, base_tools=[],
            )[0]
            delegates[role] = make_ephemeral_delegate(
                mr=mr, executor=executor, agent=agent, agent_role=role,
                max_nudges=runtime.max_nudges, max_turns=self.subagent_max_turns,
                return_diff=(role == "editor"), docker=self.docker, work_dir=work_dir,
            )

        async def _delegate_to_librarian(request: str) -> str:
            return await _lib_call(request)

        _delegate_to_librarian.__doc__ = lib_doc
        delegate_to_librarian = function_tool(
            name_override="delegate_to_librarian",
        )(_delegate_to_librarian)

        @function_tool
        async def delegate_to_editor(request: str) -> str:
            return await delegates["editor"](request)

        @function_tool
        async def delegate_to_executor(request: str) -> str:
            return await delegates["executor"](request)

        @function_tool
        async def finish(answer: str) -> str:
            await self.docker.exec_shell(
                f"cd {work_dir} && git add -A && "
                "git commit -m 'fix' 2>/dev/null || true",
            )
            return answer

        # planner (top-level)
        planner_spec = SUBAGENTS["planner"]
        planner_llm = maybe_wrap_last_n(mr.llm_model, self.last_n_observations, "planner")
        planner = Agent(
            name=planner_spec.name,
            instructions=_fmt(planner_system),
            model=planner_llm,
            tools=[delegate_to_librarian, delegate_to_editor, delegate_to_executor, finish],
            tool_use_behavior={"stop_at_tool_names": planner_spec.stop_at},
        )
        first_user_msg = (
            planner_spec.invoke_template.format(
                work_dir=work_dir, problem_statement=instruction,
            )
            if planner_spec.invoke_template else instruction
        )

        async def _execute_planner(inst, hooks):
            return await executor.execute_one(
                agent=planner, input=first_user_msg, hooks=hooks,
                max_turns=self.planner_max_turns,
            )

        print(f"[{self.label}] Running planner...")
        try:
            await invoke_agent_logged(**mr.logged(
                agent_type="planner",
                agent_obj_factory=lambda: planner, execute=_execute_planner,
                agent_id="planner_0", mode=CreateMode.CREATE,
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
