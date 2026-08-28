from eval_pipeline.agents.history.continuation import wrap_continuation_request
from eval_pipeline.agents.history.freshness import build_freshness_report
from eval_pipeline.agents.history.last_n_observation import (
    ELISION_MARKER,
    LastNObservationModel,
    elide_old_observations,
    maybe_wrap_last_n,
    resolve_per_role,
)
from eval_pipeline.agents.history.shared_subagent import SharedSubAgent
from eval_pipeline.agents.history.trim import _trim_trailing_submit
from eval_pipeline.agents.history.strategies import (
    BranchThresholdStrategy,
    CompactionContext,
    CompactionStrategy,
    DEFAULT_BRANCH_THRESHOLD_CHARS,
    KeepAllStrategy,
)
from eval_pipeline.agents.history.view_state import (
    ViewedLinesTracker,
    measure_session_new_chars,
    parse_view_action,
)

__all__ = [
    "BranchThresholdStrategy",
    "CompactionContext",
    "CompactionStrategy",
    "DEFAULT_BRANCH_THRESHOLD_CHARS",
    "ELISION_MARKER",
    "KeepAllStrategy",
    "LastNObservationModel",
    "SharedSubAgent",
    "ViewedLinesTracker",
    "_trim_trailing_submit",
    "build_freshness_report",
    "elide_old_observations",
    "maybe_wrap_last_n",
    "measure_session_new_chars",
    "parse_view_action",
    "resolve_per_role",
    "wrap_continuation_request",
]
