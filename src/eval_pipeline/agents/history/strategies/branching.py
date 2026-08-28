from __future__ import annotations

import logging

from eval_pipeline.agents.history.strategies.base import (
    CompactionContext,
    _DefaultRenderMixin,
)

logger = logging.getLogger(__name__)


DEFAULT_BRANCH_THRESHOLD_CHARS = 500


class BranchThresholdStrategy(_DefaultRenderMixin):

    def __init__(
        self,
        threshold_chars: int = DEFAULT_BRANCH_THRESHOLD_CHARS,
        *,
        agent_id: str = "shared",
    ) -> None:
        self.threshold_chars = threshold_chars
        self.agent_id = agent_id
        self.drop_count = 0
        self.keep_count = 0

    async def compact(
        self,
        history: list,
        session_delta: list,
        ctx: CompactionContext,
    ) -> list:
        new_chars = getattr(ctx, "new_chars", 0)
        if new_chars < self.threshold_chars:
            self.drop_count += 1
            logger.info(
                "%s branch DROP: new_chars=%d < %d "
                "(session discarded; drops=%d keeps=%d)",
                self.agent_id, new_chars, self.threshold_chars,
                self.drop_count, self.keep_count,
            )
            return history
        history.extend(session_delta)
        self.keep_count += 1
        logger.info(
            "%s branch KEEP: new_chars=%d >= %d "
            "(history+%d msgs; drops=%d keeps=%d)",
            self.agent_id, new_chars, self.threshold_chars,
            len(session_delta), self.drop_count, self.keep_count,
        )
        return history
