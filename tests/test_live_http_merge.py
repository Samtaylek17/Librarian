"""Folding the raw HTTP sidecar (``live_http.jsonl``) into the episode.

The trajectory's parsed ``llm_input`` is replaced by the raw ``raw_prompt``
(+ ``raw_completion``) captured at the client layer, matched by
``response_id`` and never by position, so MAS sub-agents interleaving into
one shared sidecar fold correctly. The sidecar is deleted only when every
turn that names a ``response_id`` was folded."""

from __future__ import annotations

import json
from pathlib import Path

from eval_pipeline.pipeline.episode import _merge_live_http


def _call(response_id: str | None, history_bytes: int = 4096) -> dict:
    """A Turn-shaped call dict carrying a bulky parsed ``llm_input`` and a
    ``raw_output`` message item whose ``provider_data`` names the response."""
    raw_output = []
    if response_id is not None:
        raw_output.append(
            {"type": "message", "provider_data": {"response_id": response_id}}
        )
    return {
        "llm_input": [{"role": "user", "content": "H" * history_bytes}],
        "raw_output": raw_output,
        "llm_stats": {"input_tokens": 5},
    }


def _write_sidecar(tmp_path: Path, response_ids: list[str]) -> Path:
    p = tmp_path / "live_http.jsonl"
    with open(p, "w") as f:
        for rid in response_ids:
            f.write(json.dumps({
                "response": {"id": rid},
                "raw_prompt": f"PROMPT::{rid}",
                "raw_completion": f"<tool_call>::{rid}",
            }) + "\n")
    return p


def test_matches_by_response_id_not_position(tmp_path):
    calls = [_call("AAA"), _call("BBB")]
    # sidecar order is reversed vs the calls
    p = _write_sidecar(tmp_path, ["BBB", "AAA"])

    enriched, complete = _merge_live_http(calls, p)

    assert enriched and complete
    assert calls[0]["raw_prompt"] == "PROMPT::AAA"
    assert calls[1]["raw_prompt"] == "PROMPT::BBB"
    assert calls[0]["raw_completion"] == "<tool_call>::AAA"


def test_parsed_llm_input_dropped_on_fold(tmp_path):
    calls = [_call("AAA")]
    p = _write_sidecar(tmp_path, ["AAA"])

    before = len(json.dumps(calls, default=str))
    _merge_live_http(calls, p)
    after = len(json.dumps(calls, default=str))

    assert "llm_input" not in calls[0]          # raw_prompt supersedes it
    assert calls[0]["raw_prompt"] == "PROMPT::AAA"
    assert after < before                        # the bulky history is gone


def test_partial_match_keeps_sidecar_and_unmatched_turn(tmp_path):
    calls = [_call("AAA"), _call("BBB")]
    p = _write_sidecar(tmp_path, ["AAA"])        # BBB missing

    enriched, complete = _merge_live_http(calls, p)

    assert enriched and not complete             # do NOT delete the sidecar
    assert "llm_input" in calls[1]               # unmatched turn untouched
    assert calls[1].get("raw_prompt") is None


def test_error_turn_without_id_does_not_block_deletion(tmp_path):
    calls = [_call("AAA"), _call(None)]          # 2nd is a failed/empty call
    p = _write_sidecar(tmp_path, ["AAA"])

    enriched, complete = _merge_live_http(calls, p)

    assert enriched and complete                 # no-id turn carries no raw


def test_idempotent_and_missing_sidecar(tmp_path):
    calls = [_call("AAA")]
    p = _write_sidecar(tmp_path, ["AAA"])
    _merge_live_http(calls, p)

    # second fold is a no-op (already folded)
    assert _merge_live_http(calls, p) == (False, False)
    # missing sidecar is safe
    assert _merge_live_http(calls, tmp_path / "gone.jsonl") == (False, False)
