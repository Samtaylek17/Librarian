"""Phase 5: extractors read per-call stats from the nested ``llm_stats``
of Turn dicts, and still work on legacy flat-shaped logs."""

from __future__ import annotations

from dataclasses import asdict

from eval_pipeline.agents.base import AgentResult
from eval_pipeline.agents.runtime.turn import LLMStats, Turn
from eval_pipeline.extractors.joules import JoulesExtractor
from eval_pipeline.extractors.token_usage import TokenUsageExtractor
from eval_pipeline.pipeline.schema import EpisodeData

KNOWN_MODEL = "Qwen/Qwen3.5-35B-A3B"


def _turn(agent_id="a", agent_type="vanilla", **stats) -> dict:
    return asdict(Turn(agent_id=agent_id, agent_type=agent_type,
                       llm_stats=LLMStats(**stats)))


def _result(calls) -> AgentResult:
    return AgentResult(raw_output={"llm_calls": calls}, llm_calls=calls)


def test_token_usage_from_nested_turns():
    calls = [
        _turn(input_tokens=100, output_tokens=20, cached_tokens=8),
        _turn(input_tokens=50, output_tokens=10, cached_tokens=0),
    ]
    ep = EpisodeData(model=KNOWN_MODEL)
    TokenUsageExtractor().enrich(ep, _result(calls), None)
    assert ep.token_usage is not None
    assert ep.token_usage.input_tokens == 150
    assert ep.token_usage.output_tokens == 30
    assert ep.token_usage.cached_tokens == 8
    assert ep.token_usage.total_tokens == 180


def test_token_usage_back_compat_flat():
    calls = [{"input_tokens": 100, "output_tokens": 20, "cached_tokens": 5}]
    ep = EpisodeData(model=KNOWN_MODEL)
    TokenUsageExtractor().enrich(ep, _result(calls), None)
    assert ep.token_usage.input_tokens == 100
    assert ep.token_usage.total_tokens == 120


def test_joules_from_nested_turns():
    calls = [
        _turn(joules=100.0, joules_raw=120.0, joules_per_gpu=[100.0],
              duration_seconds=2.0, input_tokens=10, output_tokens=5),
        _turn(joules=None),  # unmeasured call → skipped
    ]
    ep = EpisodeData(model=KNOWN_MODEL)
    JoulesExtractor().enrich(ep, _result(calls), None)
    assert ep.energy is not None
    assert ep.energy.total_joules == 100.0
    assert ep.energy.n_calls_measured == 1
    pc = ep.energy.details["per_call"][0]
    assert pc["agent_id"] == "a"
    assert pc["agent_type"] == "vanilla"
    assert pc["input_tokens"] == 10
