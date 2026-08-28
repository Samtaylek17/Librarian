from __future__ import annotations

from eval_pipeline.agents.history.strategies.base import (
    CompactionContext,
    _DefaultRenderMixin,
)


class KeepAllStrategy(_DefaultRenderMixin):

    def __init__(self) -> None:
        self.drop_count = 0
        self.keep_count = 0

    async def compact(
        self,
        history: list,
        session_delta: list,
        ctx: CompactionContext,
    ) -> list:
        history.extend(session_delta)
        self.keep_count += 1
        return history
