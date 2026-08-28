from __future__ import annotations

import logging
from typing import Any

from eval_pipeline.agents.history.strategies.branching import (
    BranchThresholdStrategy,
    DEFAULT_BRANCH_THRESHOLD_CHARS,
)
from eval_pipeline.agents.history.strategies.keep_all import KeepAllStrategy

logger = logging.getLogger(__name__)


REGISTERED_NAMES = (
    "branching",
    "keep_all",
)


def build_memory_strategy(
    name: str,
    *,
    agent_id: str = "shared",
    branch_threshold_chars: int = DEFAULT_BRANCH_THRESHOLD_CHARS,
) -> Any:
    name = (name or "").lower().strip()
    if name == "branching" or not name:
        return BranchThresholdStrategy(
            threshold_chars=branch_threshold_chars, agent_id=agent_id,
        )
    if name == "keep_all":
        return KeepAllStrategy()
    raise ValueError(
        f"Unknown memory_strategy name {name!r}; "
        f"valid: {', '.join(REGISTERED_NAMES)}"
    )
