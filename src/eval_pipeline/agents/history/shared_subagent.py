from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agents import Agent

from eval_pipeline.agents._reasoning import strip_reasoning
from eval_pipeline.agents.history.continuation import wrap_continuation_request
from eval_pipeline.agents.history.freshness import build_freshness_report
from eval_pipeline.agents.history.strategies import (
    BranchThresholdStrategy,
    CompactionContext,
    CompactionStrategy,
    DEFAULT_BRANCH_THRESHOLD_CHARS,
)
from eval_pipeline.agents.history.view_state import (
    ViewedLinesTracker,
    measure_session_new_chars,
)
from eval_pipeline.agents.runtime import safe_runner_run
from eval_pipeline.utils.docker import DockerExecutor

logger = logging.getLogger(__name__)


@dataclass
class SharedSubAgent:

    agent: Agent
    docker: DockerExecutor
    work_dir: str
    agent_id: str = "shared"
    branch_threshold_chars: int = DEFAULT_BRANCH_THRESHOLD_CHARS
    strategy: CompactionStrategy | None = None
    model: str | None = None
    # Ablation toggle: when False, skip the per-wake git-diff FILE FRESHNESS
    # REPORT — the continuation user-msg carries the request only.
    freshness_report: bool = True

    history: list = field(default_factory=list)
    last_shown_diff: dict[str, str] = field(default_factory=dict)
    viewed_lines: ViewedLinesTracker = field(default_factory=ViewedLinesTracker)
    last_wake_view_files: set = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.strategy is None:
            self.strategy = BranchThresholdStrategy(
                threshold_chars=self.branch_threshold_chars,
                agent_id=self.agent_id,
            )

    @property
    def drop_count(self) -> int:
        return getattr(self.strategy, "drop_count", 0)

    @property
    def keep_count(self) -> int:
        return getattr(self.strategy, "keep_count", 0)

    def state_checkpoint(self) -> dict:
        return {
            "history_len": len(self.history),
            "last_shown_diff": dict(self.last_shown_diff),
            "viewed_lines": self.viewed_lines.snapshot(),
        }

    def restore_state(self, ck: dict) -> list:
        delta = list(self.history[ck["history_len"]:])
        self.history = self.history[: ck["history_len"]]
        self.viewed_lines.restore(ck["viewed_lines"])
        return delta

    def seed_history(self, items: list, viewed_files: set[str]) -> None:
        self.history = list(items)
        for f in viewed_files:
            self.viewed_lines._fully_viewed.add(f)

    async def invoke(
        self,
        request: str,
        caller_role: str,
        *,
        hooks,
        max_turns: int | None = None,
        text_only_policy: Any | None = None,
    ) -> str:
        if self.history:
            shown_history = await self.strategy.render(self.history, request)
            if self.freshness_report:
                freshness = await build_freshness_report(
                    self.docker,
                    self.work_dir,
                    self.last_shown_diff,
                    self.viewed_lines,
                )
            else:
                freshness = ""
            wrapped = wrap_continuation_request(
                request, caller_role, freshness
            )
            input_items: list = list(shown_history) + [
                {"role": "user", "content": wrapped},
            ]
        else:
            input_items = [{"role": "user", "content": request}]

        runner_kwargs: dict[str, Any] = {"hooks": hooks}
        if max_turns is not None:
            runner_kwargs["max_turns"] = max_turns
        if text_only_policy is not None:
            runner_kwargs["text_only_policy"] = text_only_policy

        result = await safe_runner_run(
            self.agent, input_items, **runner_kwargs,
        )
        final_output = result.final_output or ""
        if self.model:
            final_output = strip_reasoning(final_output, self.model)

        full = result.to_input_list()
        delta = full[len(self.history):]
        snap = self.viewed_lines.snapshot()
        new_chars = measure_session_new_chars(delta, self.viewed_lines)
        ctx = CompactionContext(new_chars=new_chars, request=request)

        import json as _json
        from eval_pipeline.agents.history.view_state import parse_view_action
        wake_view: set[str] = set()
        for msg in delta:
            if not isinstance(msg, dict): continue
            if msg.get("type") != "function_call": continue
            try:
                args = _json.loads(msg.get("arguments", "{}"))
            except Exception:
                continue
            view = parse_view_action(msg.get("name", ""), args)
            if view is None: continue
            path = view[0]
            if path: wake_view.add(path)
        self.last_wake_view_files = wake_view

        before_drop = self.drop_count
        self.history = await self.strategy.compact(self.history, delta, ctx)
        if self.drop_count > before_drop:
            self.viewed_lines.restore(snap)
        return final_output
