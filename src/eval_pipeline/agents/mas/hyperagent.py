"""HyperAgent MAS — planner + Navigator/Editor/Executor sub-agents.

Layout:
    I.   CONFIG       — :class:`HyperAgentConfig`.
    II.  SUB-AGENTS   — :data:`SUBAGENTS` spec table. Prompt bodies live in
                        :mod:`mas.prompts._hyperagent`.
    III. TOOLS        — per-role tool cocktails (upstream-aligned: read-only
                        Navigator, edit Editor, bash-only Executor).
    IV.  HARNESS      — :class:`HyperAgentMASRunner`.

Reference (upstream prompts): github.com/FSoft-AI4Code/HyperAgent

Note: upstream specialised each sub-agent with jedi/ctags/Zoekt tools; for
the base-substitution study every sub-agent instead gets the same OH default
tool set so any delta isolates the base, not the tool set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents import Agent, function_tool

from eval_pipeline.agents.base import SubAgentSpec
from eval_pipeline.agents.bases.delegation import invoke_agent_logged
from eval_pipeline.agents.history import maybe_wrap_last_n
from eval_pipeline.agents.llm.profiles import render_tool_examples
from eval_pipeline.agents.mas._base import MASRunnerBase
from eval_pipeline.agents.mas._delegation import make_ephemeral_delegate
from eval_pipeline.agents.mas._runtime import init_mas_runtime
from eval_pipeline.agents.mas.prompts._hyperagent import (
    _EDITOR_SYSTEM,
    _EXECUTOR_SYSTEM,
    _NAVIGATOR_SYSTEM,
    _PLANNER_INSTANCE_TEMPLATE,
    _PLANNER_SYSTEM,
)
from eval_pipeline.agents.runtime import CreateMode

__all__ = ["HyperAgentConfig", "HyperAgentMASRunner", "SUBAGENTS"]


# ── I. CONFIG ────────────────────────────────────────────────────────────
@dataclass
class HyperAgentConfig:
    model: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    run_timeout: int = 3600
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "dummy"
    # Round caps relaxed: upstream HyperAgent uses autogen GroupChat
    planner_max_turns: int = 1000
    subagent_max_turns: int = 1000
    max_observation_length: int = 16000
    max_nudges: int = 2
    bash_timeout: int = 1800
    seed: int | None = None
    caveman_level: str | None = None
    last_n_observations: int | dict[str, int] | None = None


# ── II. SUB-AGENTS ───────────────────────────────────────────────────────
SUBAGENTS: dict[str, SubAgentSpec] = {
    "planner": SubAgentSpec(
        name="Planner", system_prompt=_PLANNER_SYSTEM,
        invoke_template=_PLANNER_INSTANCE_TEMPLATE, stop_at=["finish"], base_tools=[],
    ),
    "navigator": SubAgentSpec(
        name="Navigator", system_prompt=_NAVIGATOR_SYSTEM,
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
class HyperAgentMASRunner(MASRunnerBase):
    CONFIG = HyperAgentConfig
    LABEL = "HyperAgentMAS"
    CAVEMAN_ROLES = ("planner", "navigator", "editor", "executor")

    def _role_tools(self, work_dir: str, runtime, submit_result):
        """III. TOOLS — upstream-aligned per-role tool cocktails."""
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
        return {
            "navigator": [nav_bash, nav_sre, submit_result],
            "editor": [nav_bash, ed_sre, submit_result],
            "executor": [ex_bash, submit_result],
        }

    async def run_with_executor(self, instruction, work_dir, *, executor, runtime):
        assert self.docker is not None, "Call setup() first"
        mr = await init_mas_runtime(self, label=self.label)
        _cav = self.caveman_fn()
        self.note_last_n()

        def _fmt(prompt: str) -> str:
            return _cav(render_tool_examples(prompt, self.model))

        @function_tool
        async def submit_result(result: str) -> str:
            return result

        role_tools = self._role_tools(work_dir, runtime, submit_result)

        # navigator / editor / executor — ephemeral subs + delegates
        delegates: dict[str, Any] = {}
        for role in ("navigator", "editor", "executor"):
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

        @function_tool
        async def delegate_to_navigator(request: str) -> str:
            return await delegates["navigator"](request)

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
            instructions=_fmt(planner_spec.system_prompt),
            model=planner_llm,
            tools=[delegate_to_navigator, delegate_to_editor, delegate_to_executor, finish],
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
        await invoke_agent_logged(**mr.logged(
            agent_type="planner",
            agent_obj_factory=lambda: planner, execute=_execute_planner,
            agent_id="planner_0", mode=CreateMode.CREATE,
            max_nudges=runtime.max_nudges,
        ))
