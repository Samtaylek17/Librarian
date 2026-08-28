from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._turn_compat import normalize_call  # noqa: E402
from scripts.episode_meta import _collect_episode_paths, identify_method  # noqa: E402
from scripts.redundancy_analysis import (  # noqa: E402
    extract_actions, _canonical_aid,
)
from scripts.redundancy_full import _overlap  # noqa: E402


COLUMNS = [
    "matrix_root", "method", "agent_dir",
    "model", "agent_class", "task_id", "seed", "timestamp",
    "turn_idx", "agent_id",
    "input_tokens", "output_tokens",
    "cached_tokens", "cache_h_delta",
    "joules", "joules_raw", "idle_power_W_sum",
    "duration_seconds",
    "is_file_read", "is_duplicate",
]


def _round3(v):
    return round(float(v), 3) if v is not None else None


def extract_turns(path: Path) -> list[list]:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return []

    matrix_root, agent_dir, method = identify_method(path)
    model = d.get("model")
    agent_cls = d.get("agent")
    task_id = d.get("task_id")
    seed = d.get("seed")
    ts = d.get("timestamp")
    runner_class = agent_cls or ""
    calls = d.get("llm_calls") or []

    hist: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list))

    rows: list[list] = []
    for turn_idx, c in enumerate(calls):
        c = normalize_call(c)  # flatten Turn dicts to legacy field names
        aid_raw = c.get("agent_id") or "?"
        aid = _canonical_aid(aid_raw, runner_class)

        is_file_read = 0
        is_duplicate = 0
        for a in extract_actions(c):
            f = a["file"]
            if a["kind"] == "write":
                for aid2 in list(hist.keys()):
                    hist[aid2][f] = []
                continue
            is_file_read = 1
            s, e = a["start"], a["end"]
            if not is_duplicate:
                dup = any(_overlap(s, e, ps, pe)
                          for (ps, pe) in hist[aid][f])
                if not dup:
                    for other_aid, files in hist.items():
                        if other_aid == aid:
                            continue
                        if any(_overlap(s, e, ps, pe)
                               for (ps, pe) in files.get(f, [])):
                            dup = True
                            break
                if dup:
                    is_duplicate = 1
            hist[aid][f].append((s, e))

        idle = c.get("idle_power_W") or []

        rows.append([
            matrix_root, method, agent_dir,
            model, agent_cls, task_id, seed, ts,
            turn_idx, aid_raw,
            c.get("input_tokens"), c.get("output_tokens"),
            c.get("cached_tokens"), c.get("cache_h_delta"),
            _round3(c.get("joules")), _round3(c.get("joules_raw")),
            round(sum(float(x) for x in idle), 3) if idle else None,
            _round3(c.get("duration_seconds")),
            is_file_read, is_duplicate,
        ])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path,
                    help="One or more matrix log roots to walk.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output CSV path.")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) // 2),
                    help="Parallel worker processes (default: half of CPU count).")
    args = ap.parse_args()

    paths = _collect_episode_paths(tuple(args.roots))
    n_files = len(paths)
    n_rows = 0

    with args.out.open("w", newline="") as out:
        w = csv.writer(out)
        w.writerow(COLUMNS)
        if args.workers <= 1:
            for p in paths:
                for row in extract_turns(p):
                    w.writerow(row); n_rows += 1
        else:
            with Pool(processes=args.workers) as pool:
                for rows in pool.imap_unordered(extract_turns, paths,
                                                chunksize=4):
                    for row in rows:
                        w.writerow(row); n_rows += 1

    print(f"wrote {n_rows} turn rows from {n_files} episodes "
          f"(workers={args.workers}) → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
