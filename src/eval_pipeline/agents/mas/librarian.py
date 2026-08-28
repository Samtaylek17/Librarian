"""Librarian MAS — OH main agent + a persistent repo-Librarian sub-agent.

The Librarian owns code-location lookup: a single long-lived instance whose
history accumulates across calls (a memory-equipped grep/view that LOCATES,
never analyses). The main agent routes every "where does X live?" through it.

Layout:
    I.   CONFIG       — :class:`LibrarianConfig`.
    II.  SUB-AGENTS   — :data:`SUBAGENTS` (main ``agent`` + ``navigator``);
                        prompts + policies in :mod:`mas.prompts._librarian`.
    III. TOOLS        — main agent: str_replace_editor + submit + librarian-
                        routed bash + ``repo_librarian``; librarian: view-only
                        bash/editor + ``submit_result`` view side-channel.
    IV.  HARNESS      — :class:`LibrarianMASRunner`. Shared librarian wiring
                        (``_make_navigator`` + memory strategy) comes from
                        :class:`LibrarianRoleMixin`; the view side-channel and
                        delegation body from :mod:`mas._librarian_role`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agents import function_tool

from eval_pipeline.agents.base import SubAgentSpec
from eval_pipeline.agents.bases.delegation import invoke_agent_logged
from eval_pipeline.agents.history import DEFAULT_BRANCH_THRESHOLD_CHARS
from eval_pipeline.agents.mas._base import MASRunnerBase
from eval_pipeline.agents.mas._librarian_role import (
    LibrarianRoleMixin,
    make_librarian_call,
    make_view_command_submit_tool,
)
from eval_pipeline.agents.mas._runtime import init_mas_runtime
from eval_pipeline.agents.mas._tool_wrap import _wrap_tools_for_boad_template
from eval_pipeline.agents.mas.prompts._librarian import (
    _AGENT_INSTANCE,
    _AGENT_SYSTEM_PROMPT,
    _LIBRARIAN_REDIRECT_SUFFIX,
    _LIBRARIAN_SYSTEM,
    REPO_LIBRARIAN_DOC,
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
class LibrarianConfig:
    model: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    run_timeout: int = 7200
    base_url: str = ""
    api_key: str = ""
    max_iterations: int = 1000
    max_observation_length: int = 16000
    max_nudges: int = 2
    bash_timeout: int = 1800
    seed: int | None = None
    branch_threshold_chars: int = DEFAULT_BRANCH_THRESHOLD_CHARS
    drop_below_threshold: bool = True
    # Caveman overlay applies to BOTH main agent and librarian sub-agent.
    caveman_level: str | None = None


# ── II. SUB-AGENTS ───────────────────────────────────────────────────────
SUBAGENTS: dict[str, SubAgentSpec] = {
    "agent": SubAgentSpec(
        name="agent",
        system_prompt=_AGENT_SYSTEM_PROMPT,
        invoke_template=_AGENT_INSTANCE,
        stop_at=["submit"],
    ),
    "navigator": SubAgentSpec(
        name="Navigator",
        system_prompt=_LIBRARIAN_SYSTEM,
        invoke_template=None,
        stop_at=["submit_result"],
        base_tools=[],
    ),
}


# ── IV. HARNESS ──────────────────────────────────────────────────────────
class LibrarianMASRunner(LibrarianRoleMixin, MASRunnerBase):
    CONFIG = LibrarianConfig
    LABEL = "Librarian"
    USE_BASE_NAME_IN_LABEL = False
    CAVEMAN_ROLES = ("agent", "librarian")
    BLOCK_SIZE_FROM_MODEL = True  # standalone librarian derived it from the model
    LIBRARIAN_ID = "navigator"  # historical turn-log label for standalone librarian
    SUBAGENTS = SUBAGENTS

    async def run_with_executor(self, instruction, work_dir, *, executor, runtime):
        assert self.docker is not None, "Call setup() first"
        mr = await init_mas_runtime(self, label=self.label)
        _cav = self.caveman_fn()

        # librarian (navigator) sub-agent — view-only tools + view side-channel
        nav_bash = make_view_only_bash(
            self.docker, work_dir,
            max_observation_length=self.max_observation_length,
            timeout=self.bash_timeout,
        )
        nav_sre = make_view_only_str_replace_editor(
            self.docker, work_dir,
            max_observation_length=self.max_observation_length,
        )
        pending_view_output: list[str] = []
        submit_result = make_view_command_submit_tool(self.docker, pending_view_output)

        nav_spec = self.SUBAGENTS["navigator"]
        navigator_agent, _ = self.base_agent_builder(
            docker=self.docker, work_dir=work_dir, llm_model=mr.llm_model,
            agent_name=nav_spec.name, system_prompt=_cav(nav_spec.system_prompt),
            extra_tools=[nav_bash, nav_sre, submit_result],
            stop_at_tool_names=nav_spec.stop_at,
            max_observation_length=self.max_observation_length,
            bash_timeout=self.bash_timeout, base_tools=[],
        )
        shared_navigator = self._make_navigator(
            navigator_agent=navigator_agent, work_dir=work_dir,
        )

        _lib_call = make_librarian_call(
            mr=mr, shared_librarian=shared_navigator, librarian_id=self.LIBRARIAN_ID,
            caller_role="agent", max_turns=self.max_iterations,
            pending_view_output=pending_view_output, max_nudges=self.max_nudges,
            text_only_policy=NudgePolicy(max_retries=2), swallow_errors=False,
        )

        async def _repo_librarian(query: str) -> str:
            return await _lib_call(query)

        _repo_librarian.__doc__ = REPO_LIBRARIAN_DOC
        repo_librarian = function_tool(name_override="repo_librarian")(_repo_librarian)

        # main agent — librarian-routed bash + str_replace_editor + submit
        from eval_pipeline.agents.tools.policies.librarian_bash import (
            make_librarian_bash_tool,
        )
        librarian_bash = make_librarian_bash_tool(
            self.docker, work_dir,
            max_observation_length=runtime.max_observation_length,
            timeout=runtime.bash_timeout,
        )
        agent_spec = self.SUBAGENTS["agent"]
        agent, agent_tools = self.base_agent_builder(
            docker=self.docker, work_dir=work_dir, llm_model=mr.llm_model,
            agent_name=agent_spec.name, system_prompt=_cav(agent_spec.system_prompt),
            base_tools=["str_replace_editor", "submit"],
            extra_tools=[librarian_bash, repo_librarian],
            stop_at_tool_names=agent_spec.stop_at,
            max_observation_length=runtime.max_observation_length,
            bash_timeout=runtime.bash_timeout,
        )
        _wrap_tools_for_boad_template(
            agent_tools, fn_call_rules=True,
            fn_call_rules_text=_LIBRARIAN_REDIRECT_SUFFIX,
        )

        first_user_msg = agent_spec.invoke_template.format(
            work_dir=work_dir, problem_statement=instruction,
            base_commit=self.docker._initial_head or "HEAD",
        )

        async def _execute_agent(inst, hooks):
            return await executor.execute_one(
                agent=agent, input=first_user_msg, hooks=hooks,
                max_turns=runtime.max_turns,
            )

        print(
            f"[{self.label}] Running (max_turns={runtime.max_turns}, "
            f"branch_threshold={self.branch_threshold_chars}c)..."
        )
        try:
            await invoke_agent_logged(**mr.logged(
                agent_type="agent", agent_obj_factory=lambda: agent,
                execute=_execute_agent, agent_id="agent_0",
                mode=CreateMode.CREATE, max_nudges=runtime.max_nudges,
            ))
        finally:
            cleanup = getattr(shared_navigator, "close", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception as e:  # noqa: BLE001
                    logger.warning("navigator close failed: %s", e)
