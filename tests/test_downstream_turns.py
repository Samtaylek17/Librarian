"""Phase 5: downstream readers (episode_meta + the _turn_compat shim)
parse the new Turn-shaped llm_calls and still parse legacy flat logs."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from _turn_compat import normalize_call, stat  # noqa: E402

from eval_pipeline.agents.runtime.turn import (  # noqa: E402
    ExecStats, LLMStats, Observation, Turn,
)


def _turn_dict(agent_id="navigator-1", agent_type="navigator", **stats):
    t = Turn(
        agent_id=agent_id,
        agent_type=agent_type,
        llm_input=[{"role": "user", "content": "hi"}],
        llm_output_text="ok",
        parsed_actions=[{"type": "function_call", "name": "bash"}],
        llm_stats=LLMStats(**stats),
        observations=[Observation(
            tool_name="bash", args={"command": "ls"}, payload="out",
            exec_stats=ExecStats(exit_code=0, output_len=3),
            child_agent_id=None,
        )],
    )
    return asdict(t)


# ── _turn_compat ────────────────────────────────────────────────────────

def test_stat_nested_and_flat():
    assert stat(_turn_dict(input_tokens=42), "input_tokens") == 42
    assert stat({"input_tokens": 7}, "input_tokens") == 7
    assert stat({}, "joules", 0) == 0


def test_normalize_maps_turn_to_legacy_fields():
    c = normalize_call(_turn_dict(input_tokens=10, output_tokens=5,
                                  joules=99.0))
    assert c["input"] == [{"role": "user", "content": "hi"}]
    assert c["text"] == "ok"
    assert c["action"] == [{"type": "function_call", "name": "bash"}]
    assert c["agent_role"] == "navigator"
    assert c["input_tokens"] == 10 and c["joules"] == 99.0
    assert c["observation"] == [{"tool_name": "bash", "output": "out"}]


def test_normalize_passthrough_flat():
    flat = {"input_tokens": 3, "action": "x", "agent_role": "agent"}
    assert normalize_call(flat) is flat


# ── episode_meta round-trip ─────────────────────────────────────────────

def test_episode_meta_parses_turn_shaped_log(tmp_path):
    from scripts.episode_meta import extract_one

    calls = [
        _turn_dict(agent_id="agent-1", input_tokens=100, output_tokens=20,
                   cache_h_delta=4.0, duration_seconds=1.5,
                   idle_power_W=[60.0, 61.0]),
        _turn_dict(agent_id="navigator-1", agent_type="navigator",
                   input_tokens=300, output_tokens=40,
                   cache_h_delta=2.0, duration_seconds=0.5),
    ]
    episode = {
        "task_id": "django__django-1",
        "agent": "LibrarianMASRunner",
        "model": "Qwen/Qwen3.5-35B-A3B",
        "score": 1.0, "passed": True,
        "eval_raw_output": "",
        "token_usage": {"input_tokens": 400, "output_tokens": 60,
                        "cached_tokens": 6, "total_tokens": 460},
        "energy": {"total_joules": 123.0, "total_joules_clean": 120.0},
        "llm_calls": calls,
    }
    p = tmp_path / "task" / "agent_seed0" / "SWEBenchEvaluator_x.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(episode))

    rec = extract_one(p)
    assert rec["n_llm_calls"] == 2
    assert rec["input_tokens"] == 400          # from token_usage (extractor)
    assert rec["total_joules"] == 123.0        # from energy
    assert rec["max_input_tokens"] == 300      # per-call, from llm_stats
    assert round(rec["total_gpu_duration_seconds"], 1) == 2.0
    assert rec["idle_power_W"] == [60.0, 61.0]
    # per-agent breakdown keyed by agent_id, summed from nested llm_stats
    assert rec["per_agent"]["agent-1"]["input_tokens"] == 100
    assert rec["per_agent"]["navigator-1"]["input_tokens"] == 300
    assert rec["per_agent"]["navigator-1"]["n_calls"] == 1
