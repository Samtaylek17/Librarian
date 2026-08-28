from __future__ import annotations

from eval_pipeline.agents.base import AgentResult
from eval_pipeline.benchmarks.base import EvalResult
from eval_pipeline.extractors.base import Extractor, call_stat
from eval_pipeline.pipeline.schema import EpisodeData, TokenUsage


class TokenUsageExtractor(Extractor):

    def enrich(
        self,
        episode: EpisodeData,
        agent_result: AgentResult,
        eval_result: EvalResult,
    ) -> None:
        # Read the single source of truth — agent_result.llm_calls — the
        # same field joules.py uses. (raw_output no longer carries a
        # duplicate copy of the trajectory.)
        if agent_result.llm_calls:
            episode.token_usage = self._from_llm_calls(agent_result.llm_calls)

    def _from_llm_calls(self, calls: list[dict]) -> TokenUsage | None:
        if not calls:
            return None

        total_input = sum(call_stat(c, "input_tokens", 0) for c in calls)
        total_output = sum(call_stat(c, "output_tokens", 0) for c in calls)
        total_cached = sum(call_stat(c, "cached_tokens", 0) for c in calls)
        total_theoretical_cached = sum(
            call_stat(c, "theoretical_cached_tokens", 0) for c in calls
        )

        return TokenUsage(
            input_tokens=total_input,
            output_tokens=total_output,
            cached_tokens=total_cached,
            theoretical_cached_tokens=total_theoretical_cached,
            total_tokens=total_input + total_output,
        )
