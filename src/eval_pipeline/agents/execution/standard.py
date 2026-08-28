from __future__ import annotations

import asyncio
import time
from typing import Any

from eval_pipeline.agents.runtime import safe_runner_run


class StandardExecution:

    def __init__(self, total_timeout: float) -> None:
        self.total_timeout = total_timeout
        self.start = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def remaining(self) -> float:
        return max(0.0, self.total_timeout - self.elapsed())

    async def execute_one(
        self,
        *,
        agent: Any,
        input: Any,
        hooks: Any = None,
        text_only_policy: Any | None = None,
        max_turns: int | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        if hooks is not None:
            kwargs["hooks"] = hooks
        if text_only_policy is not None:
            kwargs["text_only_policy"] = text_only_policy
        if max_turns is not None:
            kwargs["max_turns"] = max_turns
        return await asyncio.wait_for(
            safe_runner_run(agent, input, **kwargs),
            timeout=self.remaining(),
        )
