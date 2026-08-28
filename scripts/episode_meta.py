"""Stream-extract scalar episode metadata from SWEBenchEvaluator_*.json files."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from decimal import Decimal
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterator

import ijson


_SEED_RE = re.compile(r"_seed\d+$")


def identify_method(path: Path) -> tuple[str, str, str]:
    agent_dir = path.parent.name
    method = _SEED_RE.sub("", agent_dir)
    matrix_root = path.parent.parent.parent.name
    return matrix_root, agent_dir, method


_WANT_SCALARS = {
    "task_id", "benchmark", "agent", "model", "agent_mode", "seed",
    "timestamp",
    "score", "passed",
    "agent_duration_seconds", "total_duration_seconds",
    "agent_error",
}

_WANT_NESTED = {
    "token_usage.input_tokens": "input_tokens",
    "token_usage.output_tokens": "output_tokens",
    "token_usage.cached_tokens": "cached_tokens",
    "token_usage.total_tokens": "total_tokens",
    "energy.total_joules": "total_joules",
    "energy.total_joules_clean": "total_joules_clean",
}


def _detect_eval_error(eval_raw_output: str) -> str | None:
    if not eval_raw_output:
        return None
    if eval_raw_output.startswith("Harness failed"):
        if "ConnectionError" in eval_raw_output or "Name or service not known" in eval_raw_output:
            return "harness_dns_error"
        if "Max retries exceeded" in eval_raw_output:
            return "harness_network_error"
        return "harness_subprocess_error"
    if eval_raw_output.startswith("Harness timed out"):
        return "harness_timeout"
    if eval_raw_output.startswith("Report not found"):
        return "harness_no_report"
    if "ConnectionError" in eval_raw_output and "raw.githubusercontent.com" in eval_raw_output:
        return "harness_dns_error"
    return None


def _classify_failure_mode(agent_error: str | None) -> str | None:
    if not agent_error:
        return None
    core = agent_error
    while " | reeval" in core:
        core = core.split(" | reeval", 1)[0]
    core = core.strip()
    if not core:
        return None
    if core.startswith("run_timeout") or core.startswith("TimeoutError"):
        return "timeout"
    if core.startswith("context_window_exceeded"):
        return "context_limit"
    if core.startswith("StuckTerminate"):
        return "stuck"
    return "other_error"


def extract_one(path: Path) -> dict[str, Any] | None:
    matrix_root, agent_dir, method = identify_method(path)
    rec: dict[str, Any] = {
        "_path": str(path),
        "matrix_root": matrix_root,
        "agent_dir": agent_dir,
        "method": method,
        "eval_error": None,
    }
    n_llm_calls = 0
    max_input_tokens = 0
    cache_h_tokens = 0
    total_gpu_duration = 0.0
    idle_power_W: list[float] | None = None
    per_agent: dict[str, dict[str, float]] = {}

    cur_aid: str | None = None
    cur_in = 0
    cur_out = 0
    cur_cache_h = 0.0
    cur_dur = 0.0
    cur_idle: list[float] = []
    in_idle_list = False

    try:
        with open(path, "rb") as f:
            for prefix, event, value in ijson.parse(f):
                # Per-call stats moved under a nested ``llm_stats`` map when
                # entries became Turn dicts; normalise so both the old flat
                # shape and the new nested shape parse identically.
                np = prefix.replace(
                    "llm_calls.item.llm_stats.", "llm_calls.item.",
                )
                if prefix == "llm_calls.item":
                    if event == "start_map":
                        n_llm_calls += 1
                        cur_aid = None
                        cur_in = cur_out = 0
                        cur_cache_h = cur_dur = 0.0
                        cur_idle = []
                        in_idle_list = False
                    elif event == "end_map":
                        aid = cur_aid or "?"
                        a = per_agent.setdefault(aid, {
                            "input_tokens": 0, "output_tokens": 0,
                            "cache_h_tokens": 0, "n_calls": 0,
                            "gpu_duration_seconds": 0.0,
                        })
                        a["input_tokens"]  += cur_in
                        a["output_tokens"] += cur_out
                        a["cache_h_tokens"] += int(cur_cache_h)
                        a["n_calls"] += 1
                        a["gpu_duration_seconds"] += cur_dur
                        if cur_in > max_input_tokens:
                            max_input_tokens = cur_in
                        cache_h_tokens += int(cur_cache_h)
                        total_gpu_duration += cur_dur
                        if idle_power_W is None and cur_idle:
                            idle_power_W = cur_idle
                    continue

                if np == "llm_calls.item.idle_power_W":
                    if event == "start_array":
                        in_idle_list = True
                    elif event == "end_array":
                        in_idle_list = False
                    continue
                if in_idle_list and np == "llm_calls.item.idle_power_W.item" \
                        and event == "number":
                    cur_idle.append(float(value) if isinstance(value, Decimal) else float(value))
                    continue

                if np == "llm_calls.item.agent_id" and event == "string":
                    cur_aid = value
                    continue
                if event == "number":
                    if isinstance(value, Decimal):
                        value = float(value)
                    if np == "llm_calls.item.input_tokens":
                        cur_in = int(value); continue
                    if np == "llm_calls.item.output_tokens":
                        cur_out = int(value); continue
                    if np == "llm_calls.item.cache_h_delta":
                        cur_cache_h = float(value); continue
                    if np == "llm_calls.item.duration_seconds":
                        cur_dur = float(value); continue

                if event in ("string", "number", "boolean", "null"):
                    if prefix == "eval_raw_output" and isinstance(value, str):
                        rec["eval_error"] = _detect_eval_error(value)
                        continue
                    if isinstance(value, Decimal):
                        value = float(value)
                    if prefix in _WANT_SCALARS:
                        rec[prefix] = value
                    elif prefix in _WANT_NESTED:
                        rec[_WANT_NESTED[prefix]] = value
    except (ijson.JSONError, OSError) as e:
        rec["_error"] = f"{type(e).__name__}: {e}"
        return rec

    rec["n_llm_calls"] = n_llm_calls
    rec["max_input_tokens"] = max_input_tokens or None
    rec["cache_h_tokens"] = cache_h_tokens
    rec["total_gpu_duration_seconds"] = round(total_gpu_duration, 3)
    rec["idle_power_W"] = idle_power_W or []
    rec["per_agent"] = per_agent
    ae = rec.get("agent_error")
    if isinstance(ae, str) and len(ae) > 500:
        rec["agent_error"] = ae[:500] + "…"
    rec["failure_mode"] = _classify_failure_mode(rec.get("agent_error"))
    return rec


def _collect_episode_paths(roots: tuple[Path, ...]) -> list[Path]:
    by_dir: dict[Path, Path] = {}
    for root in roots:
        for p in Path(root).rglob("*Evaluator_*.json"):
            prev = by_dir.get(p.parent)
            if prev is None or p.name > prev.name:
                by_dir[p.parent] = p
    return sorted(by_dir.values())


def iter_episode_meta(*roots: Path, workers: int = 1) -> Iterator[dict[str, Any]]:
    paths = _collect_episode_paths(tuple(roots))
    if workers <= 1:
        for p in paths:
            r = extract_one(p)
            if r is not None:
                yield r
        return
    with Pool(processes=workers) as pool:
        for r in pool.imap_unordered(extract_one, paths, chunksize=8):
            if r is not None:
                yield r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path,
                    help="One or more log roots to walk.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write JSONL here instead of stdout.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="Parallel worker processes (default: half of CPU count).")
    args = ap.parse_args()

    out = open(args.out, "w") if args.out else sys.stdout
    n = 0
    for rec in iter_episode_meta(*args.roots, workers=args.workers):
        out.write(json.dumps(rec, default=str) + "\n")
        n += 1
    if args.out:
        out.close()
    print(f"extracted: {n} episodes (workers={args.workers})", file=sys.stderr)


if __name__ == "__main__":
    main()
