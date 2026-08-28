from __future__ import annotations

from typing import Any, Protocol


class ExecutionStrategy(Protocol):

    async def execute_one(
        self,
        *,
        agent: Any,
        input: Any,
        hooks: Any = None,
        text_only_policy: Any | None = None,
        max_turns: int | None = None,
    ) -> Any: ...
