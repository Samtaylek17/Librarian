from __future__ import annotations

from eval_pipeline.agents.base import AgentResult
from eval_pipeline.benchmarks.base import EvalResult
from eval_pipeline.extractors.base import Extractor, call_stat
from eval_pipeline.pipeline.schema import EpisodeData, EnergyResult


class JoulesExtractor(Extractor):
    def enrich(
        self,
        episode: EpisodeData,
        agent_result: AgentResult,
        eval_result: EvalResult,
    ) -> None:
        if not agent_result.llm_calls:
            return

        total_J = 0.0
        total_J_raw = 0.0
        n_measured = 0
        per_call: list[dict] = []
        per_gpu_total: list[float] = []
        per_gpu_total_raw: list[float] = []

        for call in agent_result.llm_calls:
            j = call_stat(call, "joules")
            if j is None:
                continue
            j = float(j)
            j_raw = call_stat(call, "joules_raw")
            j_raw = float(j_raw) if j_raw is not None else j
            per_gpu = [float(x) for x in (call_stat(call, "joules_per_gpu") or [])]
            per_gpu_raw = [
                float(x) for x in (call_stat(call, "joules_per_gpu_raw") or [])
            ]
            dur = call_stat(call, "duration_seconds")

            if per_gpu and not per_gpu_total:
                per_gpu_total = [0.0] * len(per_gpu)
            for i, x in enumerate(per_gpu[: len(per_gpu_total)]):
                per_gpu_total[i] += x
            if per_gpu_raw and not per_gpu_total_raw:
                per_gpu_total_raw = [0.0] * len(per_gpu_raw)
            for i, x in enumerate(per_gpu_raw[: len(per_gpu_total_raw)]):
                per_gpu_total_raw[i] += x

            total_J += j
            total_J_raw += j_raw
            if j > 0:
                n_measured += 1
            per_call.append({
                "agent_id": call.get("agent_id"),
                "agent_type": call.get("agent_type"),
                "joules": j,
                "joules_raw": j_raw,
                "joules_per_gpu": per_gpu,
                "joules_per_gpu_raw": per_gpu_raw,
                "idle_power_W": [
                    float(x) for x in (call_stat(call, "idle_power_W") or [])
                ],
                "duration_seconds": dur,
                "input_tokens": call_stat(call, "input_tokens"),
                "output_tokens": call_stat(call, "output_tokens"),
            })

        if not per_call:
            return

        episode.energy = EnergyResult(
            total_joules=round(total_J, 3),
            total_joules_clean=round(total_J, 3),
            n_outlier_calls=0,
            outlier_joules=0.0,
            outlier_threshold_seconds=0.0,
            n_calls_measured=n_measured,
            model_name=episode.model,
            details={
                "total_joules_raw": round(total_J_raw, 3),
                "total_joules_per_gpu": [
                    round(x, 3) for x in per_gpu_total
                ],
                "total_joules_per_gpu_raw": [
                    round(x, 3) for x in per_gpu_total_raw
                ],
                "n_calls_total": len(agent_result.llm_calls),
                "per_call": per_call,
            },
        )
