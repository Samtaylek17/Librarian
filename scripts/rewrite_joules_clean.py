#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _turn_compat import normalize_call  # noqa: E402

OUTLIER_SECONDS = 590.0


def recompute(calls: list[dict]) -> dict | None:
    total_J = 0.0
    clean_J = 0.0
    outlier_J = 0.0
    n_meas = 0
    n_out = 0
    per_gpu_total: list[float] = []
    per_gpu_clean: list[float] = []
    any_joules = False
    per_call_flags: list[tuple[int, bool]] = []

    for idx, c in enumerate(calls or []):
        j = c.get("joules")
        if j is None:
            continue
        any_joules = True
        j = float(j)
        per_gpu = [float(x) for x in (c.get("joules_per_gpu") or [])]
        dur = c.get("duration_seconds")
        is_out = dur is not None and float(dur) > OUTLIER_SECONDS

        if per_gpu and not per_gpu_total:
            per_gpu_total = [0.0] * len(per_gpu)
            per_gpu_clean = [0.0] * len(per_gpu)
        for i, x in enumerate(per_gpu[: len(per_gpu_total)]):
            per_gpu_total[i] += x
            if not is_out:
                per_gpu_clean[i] += x

        total_J += j
        if is_out:
            n_out += 1
            outlier_J += j
        else:
            clean_J += j
        if j > 0:
            n_meas += 1
        per_call_flags.append((idx, is_out))

    if not any_joules:
        return None
    return {
        "total_joules": round(total_J, 3),
        "total_joules_clean": round(clean_J, 3),
        "n_outlier_calls": n_out,
        "outlier_joules": round(outlier_J, 3),
        "outlier_threshold_seconds": OUTLIER_SECONDS,
        "n_calls_measured": n_meas,
        "total_joules_per_gpu": [round(x, 3) for x in per_gpu_total],
        "total_joules_per_gpu_clean": [round(x, 3) for x in per_gpu_clean],
        "per_call_flags": per_call_flags,
    }


def process_one(path: Path, dry_run: bool) -> str:
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return f"error ({e})"
    raw_calls = data.get("llm_calls")
    if not raw_calls:
        return "skipped_no_calls"
    calls = [normalize_call(c) for c in raw_calls]
    result = recompute(calls)
    if result is None:
        return "skipped_no_joules"

    existing = data.get("energy") or {}
    changed = (
        existing.get("total_joules_clean") != result["total_joules_clean"]
        or existing.get("n_outlier_calls") != result["n_outlier_calls"]
        or existing.get("outlier_joules") != result["outlier_joules"]
    )
    if not changed and "outlier_threshold_seconds" in existing:
        return "skipped_unchanged"

    new_energy = dict(existing)
    new_energy.update({
        "total_joules": result["total_joules"],
        "total_joules_clean": result["total_joules_clean"],
        "n_outlier_calls": result["n_outlier_calls"],
        "outlier_joules": result["outlier_joules"],
        "outlier_threshold_seconds": result["outlier_threshold_seconds"],
        "n_calls_measured": result["n_calls_measured"],
    })
    details = dict(new_energy.get("details") or {})
    if result["total_joules_per_gpu"]:
        details["total_joules_per_gpu"] = result["total_joules_per_gpu"]
        details["total_joules_per_gpu_clean"] = result["total_joules_per_gpu_clean"]
    if "per_call" in details and isinstance(details["per_call"], list):
        flags = dict(result["per_call_flags"])
        new_per_call = []
        for idx, c in enumerate(calls):
            j = c.get("joules")
            if j is None:
                continue
            dur = c.get("duration_seconds")
            is_out = dur is not None and float(dur) > OUTLIER_SECONDS
            new_per_call.append({
                "agent_id": c.get("agent_id"),
                "joules": float(j),
                "joules_per_gpu": [
                    float(x) for x in (c.get("joules_per_gpu") or [])
                ],
                "duration_seconds": dur,
                "input_tokens": c.get("input_tokens"),
                "output_tokens": c.get("output_tokens"),
                "is_outlier": is_out,
            })
        details["per_call"] = new_per_call
        details["n_calls_total"] = len(calls)
    new_energy["details"] = details

    data["energy"] = new_energy
    if dry_run:
        return "would_update"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)
    return "updated"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", help="Log directories to walk.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    from collections import Counter
    stats: Counter = Counter()
    errors: list[tuple[Path, str]] = []

    import os
    seen: set[Path] = set()
    for root in args.roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f"[WARN] {root_path} does not exist; skipping.", file=sys.stderr)
            continue
        for dirpath, _, filenames in os.walk(root_path, followlinks=True):
            for f in filenames:
                if not ("Evaluator_" in f and f.endswith(".json")):
                    continue
                real = (Path(dirpath) / f).resolve()
                if real in seen:
                    continue
                seen.add(real)
                status = process_one(real, args.dry_run)
                stats[status] += 1
                if status.startswith("error"):
                    errors.append((real, status))
                elif status in ("updated", "would_update") and stats[status] % 25 == 0:
                    print(f"  ... {stats[status]} so far ({status})", flush=True)

    print("\n=== summary ===")
    for k in sorted(stats):
        print(f"  {k:<22} {stats[k]}")
    if errors:
        print("\n=== errors (first 10) ===")
        for p_, s in errors[:10]:
            print(f"  {p_}  →  {s}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
