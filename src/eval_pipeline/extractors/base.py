from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from eval_pipeline.agents.base import AgentResult
from eval_pipeline.benchmarks.base import EvalResult
from eval_pipeline.pipeline.schema import EpisodeData


def call_stat(call: dict, key: str, default: Any = None) -> Any:
    """Read a per-call LLM statistic from a trajectory entry.

    Trajectory entries are now :class:`Turn` dicts, where measurements
    live under a nested ``llm_stats`` mapping. Older logs stored the same
    fields flat at the top level. This reads the nested value when present
    and falls back to the flat one, so both shapes work.
    """
    stats = call.get("llm_stats")
    if isinstance(stats, dict) and key in stats:
        return stats.get(key, default)
    return call.get(key, default)


class Extractor(ABC):

    @abstractmethod
    def enrich(
        self,
        episode: EpisodeData,
        agent_result: AgentResult,
        eval_result: EvalResult,
    ) -> None:
        ...
