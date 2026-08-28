"""HyperAgent MAS variant — the Librarian's core design applied to the
Executor role.

:class:`HyperAgentExecLibrarianMASRunner` (``hyperagent_exec_librarian``)
answers "does the Librarian generalize beyond code navigation?": the
HyperAgent baseline with ONLY the Executor swapped for the persistent
exec-librarian (Navigator / Editor untouched, ephemeral).

The exec-librarian carries over ONLY the repo-librarian's persistence
mechanism: one Executor session per episode (:class:`SharedSubAgent`,
``UPSERT``, ``keep_all`` history — the librarian's viewed-chars pruning
would drop every command-only invocation). Everything else is the verbatim
upstream Executor: same ``bash`` + ``submit_result(result)`` contract (no
``run_commands`` pointer channel), no freshness report. The planner's
5-step plan stays byte-identical to its host; only a persistence-focused
executor policy section, the ``delegate_to_executor`` doc, and the
appended Persistence section of the Executor system prompt change.

Layout (mirrors :mod:`mas.hyperagent`):
    I.   CONFIG       — :class:`HyperAgentExecLibrarianConfig`.
    II.  SUB-AGENTS   — :data:`SUBAGENTS`.
    III. TOOLS        — executor: bash + run-command submit; others as host.
    IV.  HARNESS      — the runner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agents import Agent, function_tool

from eval_pipeline.agents.base import SubAgentSpec
from eval_pipeline.agents.bases.delegation import invoke_agent_logged
from eval_pipeline.agents.history import SharedSubAgent, maybe_wrap_last_n
from eval_pipeline.agents.history.strategies import build_memory_strategy
from eval_pipeline.agents.llm.profiles import render_tool_examples
from eval_pipeline.agents.mas._base import MASRunnerBase
from eval_pipeline.agents.mas._delegation import make_ephemeral_delegate
from eval_pipeline.agents.mas._exec_librarian_role import (
    make_persistent_executor_call,
)
from eval_pipeline.agents.mas._runtime import init_mas_runtime
from eval_pipeline.agents.mas.prompts._exec_librarian import (
    _EXEC_LIBRARIAN_SYSTEM,
    _PLANNER_SYSTEM_EXEC,
    DELEGATE_TO_EXECUTOR_DOC,
)
from eval_pipeline.agents.mas.prompts._hyperagent import (
    _EDITOR_SYSTEM,
    _NAVIGATOR_SYSTEM,
    _PLANNER_INSTANCE_TEMPLATE,
)
from eval_pipeline.agents.runtime import CreateMode

logger = logging.getLogger(__name__)


# ── I. CONFIG ────────────────────────────────────────────────────────────
@dataclass
class HyperAgentExecLibrarianConfig:
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
    caveman_level: str | None = None
    last_n_observations: int | dict[str, int] | None = None


# ── II. SUB-AGENTS ───────────────────────────────────────────────────────
SUBAGENTS: dict[str, SubAgentSpec] = {
    "planner": SubAgentSpec(
        name="Planner", system_prompt=_PLANNER_SYSTEM_EXEC,
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
        name="Executor", system_prompt=_EXEC_LIBRARIAN_SYSTEM,
        invoke_template=None, stop_at=["submit_result"], base_tools=[],
    ),
}


# ── III/IV. shared executor wiring + harnesses ───────────────────────────
class _ExecLibrarianMixin:
    """Persistent-Executor wiring for the exec-librarian."""

    EXECUTOR_ID = "executor"

    def _make_persistent_executor(
        self, *, mr, work_dir, runtime, exec_system, ex_bash,
    ):
        """Build (delegation body, SharedSubAgent) for the exec-librarian."""

        @function_tool(name_override="submit_result")
        async def exec_submit(result: str) -> str:
            return result

        exec_llm = maybe_wrap_last_n(
            mr.llm_model, self.last_n_observations, "executor",
        )
        executor_agent = self.base_agent_builder(
            docker=self.docker, work_dir=work_dir, llm_model=exec_llm,
            agent_name="Executor", system_prompt=exec_system,
            extra_tools=[ex_bash, exec_submit],
            stop_at_tool_names=["submit_result"],
            max_observation_length=runtime.max_observation_length,
            bash_timeout=runtime.bash_timeout, base_tools=[],
        )[0]
        shared_executor = SharedSubAgent(
            agent=executor_agent,
            docker=self.docker,
            work_dir=work_dir,
            agent_id=self.EXECUTOR_ID,
            strategy=build_memory_strategy(
                "keep_all", agent_id=self.EXECUTOR_ID,
            ),
            model=self.model,
            freshness_report=False,
        )
        exec_call = make_persistent_executor_call(
            mr=mr, shared_executor=shared_executor,
            executor_id=self.EXECUTOR_ID, caller_role="planner",
            max_turns=self.subagent_max_turns,
            max_nudges=runtime.max_nudges,
        )
        return exec_call, shared_executor

    async def _run_planner(self, *, mr, executor, runtime, planner_spec,
                           planner_system, tools, instruction, work_dir):
        planner_llm = maybe_wrap_last_n(
            mr.llm_model, self.last_n_observations, "planner",
        )
        planner = Agent(
            name=planner_spec.name,
            instructions=planner_system,
            model=planner_llm,
            tools=tools,
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


class HyperAgentExecLibrarianMASRunner(_ExecLibrarianMixin, MASRunnerBase):
    """HyperAgent with ONLY the Executor swapped for the exec-librarian."""

    CONFIG = HyperAgentExecLibrarianConfig
    LABEL = "HyperAgentExecLibrarianMAS"
    CAVEMAN_ROLES = ("planner", "navigator", "editor", "executor")
    SUBAGENTS = SUBAGENTS

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

        # navigator / editor — ephemeral (baseline wiring)
        role_tools = {
            "navigator": [nav_bash, nav_sre, submit_result],
            "editor": [nav_bash, ed_sre, submit_result],
        }
        delegates: dict[str, Any] = {}
        for role in ("navigator", "editor"):
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

        # executor — persistent exec-librarian
        _exec_call, shared_executor = self._make_persistent_executor(
            mr=mr, work_dir=work_dir, runtime=runtime,
            exec_system=_fmt(SUBAGENTS["executor"].system_prompt),
            ex_bash=ex_bash,
        )

        @function_tool
        async def delegate_to_navigator(request: str) -> str:
            return await delegates["navigator"](request)

        @function_tool
        async def delegate_to_editor(request: str) -> str:
            return await delegates["editor"](request)

        async def _delegate_to_executor(request: str) -> str:
            return await _exec_call(request)

        _delegate_to_executor.__doc__ = DELEGATE_TO_EXECUTOR_DOC
        delegate_to_executor = function_tool(
            name_override="delegate_to_executor",
        )(_delegate_to_executor)

        @function_tool
        async def finish(answer: str) -> str:
            await self.docker.exec_shell(
                f"cd {work_dir} && git add -A && "
                "git commit -m 'fix' 2>/dev/null || true",
            )
            return answer

        await self._run_planner(
            mr=mr, executor=executor, runtime=runtime,
            planner_spec=SUBAGENTS["planner"],
            planner_system=_fmt(SUBAGENTS["planner"].system_prompt),
            tools=[delegate_to_navigator, delegate_to_editor,
                   delegate_to_executor, finish],
            instruction=instruction, work_dir=work_dir,
        )
