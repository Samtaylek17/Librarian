"""BOAD MAS — paper's best k=2 setup (orchestrator + 2 subagents + plan).

Reproduces BOAD's best SWE-bench Verified config (paper Appendix A.2.1):
the bandit-discovered top-2 sub-agents (``issue_analyzer`` and
``code_navigator``) wired to a customized orchestrator running a fixed plan.

Layout:
    I.   CONFIG       — :class:`BOADConfig` defaults (paper-aligned).
    II.  SUB-AGENTS   — :data:`SUBAGENTS` spec table. Prompt bodies live in
                        :mod:`mas.prompts._boad`.
    III. TOOLS        — what each sub-agent role gets (see ``base_tools``).
    IV.  HARNESS      — :class:`BOADMASRunner`: build 2 ephemeral subs →
                        run orchestrator on the fixed plan.

References:
  Paper:  arXiv:2512.23631 §5.2, A.1.3, A.2.1
  Repo:   github.com/iamxjy/BOAD-SWE-Agent
"""

from __future__ import annotations

from dataclasses import dataclass

from agents import function_tool

from eval_pipeline.agents.base import SubAgentSpec
from eval_pipeline.agents.bases.delegation import invoke_agent_logged
from eval_pipeline.agents.history import maybe_wrap_last_n
from eval_pipeline.agents.mas._base import MASRunnerBase
from eval_pipeline.agents.mas._delegation import make_ephemeral_delegate
from eval_pipeline.agents.mas._runtime import init_mas_runtime
from eval_pipeline.agents.mas._tool_wrap import _wrap_tools_for_boad_template
from eval_pipeline.agents.mas.prompts._boad import (
    _CODE_NAVIGATOR_DOCSTRING,
    _CODE_NAVIGATOR_INSTANCE,
    _FIXED_PLAN,
    _ISSUE_ANALYZER_DOCSTRING,
    _ISSUE_ANALYZER_INSTANCE,
    _ORCHESTRATOR_INSTANCE,
    _ORCHESTRATOR_SYSTEM,
    _SUBAGENT_SYSTEM_TEMPLATE,
)
from eval_pipeline.agents.runtime import CreateMode, NudgePolicy

# re-exported for the boad_librarian overlay (imports the wrapper + prompts)
__all__ = ["BOADConfig", "BOADMASRunner", "SUBAGENTS"]


# ── I. CONFIG ────────────────────────────────────────────────────────────
@dataclass
class BOADConfig:
    model: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    run_timeout: int = 3600
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "dummy"
    max_iterations: int = 1000
    max_observation_length: int = 30000
    max_nudges: int = 2
    bash_timeout: int = 1800
    seed: int | None = None
    caveman_level: str | None = None
    last_n_observations: int | dict[str, int] | None = None


# ── II. SUB-AGENTS ───────────────────────────────────────────────────────
_SUBAGENTS_OVERVIEW = (
    f"- name: issue_analyzer -- {_ISSUE_ANALYZER_DOCSTRING}\n"
    f"- name: code_navigator -- {_CODE_NAVIGATOR_DOCSTRING}"
)

SUBAGENTS: dict[str, SubAgentSpec] = {
    "orchestrator": SubAgentSpec(
        name="orchestrator",
        system_prompt=_ORCHESTRATOR_SYSTEM,
        invoke_template=_ORCHESTRATOR_INSTANCE,
        stop_at=["submit"],
        base_tools=None,
    ),
    "issue_analyzer": SubAgentSpec(
        name="issue_analyzer",
        system_prompt=_SUBAGENT_SYSTEM_TEMPLATE,
        invoke_template=_ISSUE_ANALYZER_INSTANCE,
        stop_at=["submit_subagent"],
        base_tools=["bash", "str_replace_editor"],
    ),
    "code_navigator": SubAgentSpec(
        name="code_navigator",
        system_prompt=_SUBAGENT_SYSTEM_TEMPLATE,
        invoke_template=_CODE_NAVIGATOR_INSTANCE,
        stop_at=["submit_subagent"],
        base_tools=["bash", "str_replace_editor"],
    ),
}


# ── IV. HARNESS ──────────────────────────────────────────────────────────
class BOADMASRunner(MASRunnerBase):
    CONFIG = BOADConfig
    LABEL = "BOADMAS"
    CAVEMAN_ROLES = ("orchestrator", "issue_analyzer", "code_navigator")

    async def run_with_executor(self, instruction, work_dir, *, executor, runtime):
        assert self.docker is not None, "Call setup() first"
        mr = await init_mas_runtime(self, label=self.label)
        _cav = self.caveman_fn()
        self.note_last_n()

        plan = _FIXED_PLAN
        print(f"[{self.label}] Using fixed plan ({len(plan)} chars)")

        @function_tool
        async def submit_subagent(result: str) -> str:
            return result

        def _build_sub(role, *, sub_role, main_task):
            """Build an edit-capable ephemeral sub-agent + its delegate tool."""
            spec = SUBAGENTS[role]
            role_llm = maybe_wrap_last_n(mr.llm_model, self.last_n_observations, role)
            agent, tools = self.base_agent_builder(
                docker=self.docker, work_dir=work_dir, llm_model=role_llm,
                agent_name=spec.name,
                system_prompt=_cav(spec.system_prompt.format(
                    role=sub_role, main_task=main_task,
                    problem_statement=instruction,
                )),
                extra_tools=[submit_subagent], stop_at_tool_names=spec.stop_at,
                max_observation_length=runtime.max_observation_length,
                bash_timeout=runtime.bash_timeout, base_tools=spec.base_tools,
            )
            _wrap_tools_for_boad_template(tools, fn_call_rules=False)
            delegate = make_ephemeral_delegate(
                mr=mr, executor=executor, agent=agent, agent_role=role,
                max_nudges=runtime.max_nudges,
                format_input=lambda c, t=spec.invoke_template: t.format(context=c),
                text_only_policy=NudgePolicy(max_retries=2),
            )
            return delegate

        ia_delegate = _build_sub(
            "issue_analyzer", sub_role="issue analysis",
            main_task="analyze issue descriptions and extract key information",
        )
        cn_delegate = _build_sub(
            "code_navigator", sub_role="code navigation",
            main_task="explore and map relevant code structure in repositories",
        )

        @function_tool
        async def issue_analyzer(context: str) -> str:
            return await ia_delegate(context)

        @function_tool
        async def code_navigator(context: str) -> str:
            return await cn_delegate(context)

        # orchestrator (top-level)
        orch_spec = SUBAGENTS["orchestrator"]
        orch_llm = maybe_wrap_last_n(mr.llm_model, self.last_n_observations, "orchestrator")
        orchestrator, orch_tools = self.base_agent_builder(
            docker=self.docker, work_dir=work_dir, llm_model=orch_llm,
            agent_name=orch_spec.name,
            system_prompt=_cav(orch_spec.system_prompt.format(
                subagents_overview=_SUBAGENTS_OVERVIEW,
            )),
            extra_tools=[issue_analyzer, code_navigator],
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
        await invoke_agent_logged(**mr.logged(
            agent_type="orchestrator",
            agent_obj_factory=lambda: orchestrator, execute=_execute_orch,
            agent_id="orchestrator_0", mode=CreateMode.CREATE,
            max_nudges=runtime.max_nudges,
        ))
