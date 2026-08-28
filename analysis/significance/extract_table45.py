#!/usr/bin/env python
"""Extract tidy per-episode records for Table 4/5 significance testing."""
import sys, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from analysis._common import log_dir  # noqa: E402
from scripts.episode_meta import extract_one, _collect_episode_paths  # noqa: E402

DEFAULT_OUT = Path(_REPO) / "analysis" / "data" / "table45_episodes.jsonl"
DIRS = {
    "35B-A3B": ["qwen36_35b_a3b_500_3agents", "caveman_500_2agents_seed0"],
    "27B":     ["qwen36_27b_fp8_500_5agents", "caveman_lastn_27b_500_6agents_seed0"],
}
# method (agent_dir sans _seed) -> (family, label)
METHOD_MAP = {
    "hyperagent":                 ("HyperAgent", "HyperAgent"),
    "hyperagent_librarian":       ("HyperAgent", "+Librarian"),
    "hyperagent_lastn":           ("HyperAgent", "+LastNObservation"),
    "hyperagent_caveman":         ("HyperAgent", "+Caveman"),
    "hyperagent_librarian_caveman":("HyperAgent","+Caveman+Librarian"),
    "boad":                       ("BOAD", "BOAD"),
    "boad_librarian":             ("BOAD", "+Librarian"),
    "boad_lastn":                 ("BOAD", "+LastNObservation"),
    "boad_caveman":               ("BOAD", "+Caveman"),
    "boad_librarian_caveman":     ("BOAD", "+Caveman+Librarian"),
}

def one(args):
    model, path = args
    rec = extract_one(Path(path))
    if rec is None:
        return None
    fam = METHOD_MAP.get(rec.get("method"))
    if fam is None:
        return None
    family, label = fam
    passed = rec.get("passed")
    n = rec.get("n_llm_calls", 0)
    valid = (rec.get("eval_error") is None) and (n and n > 0) and (passed is not None)
    return {
        "model": model,
        "family": family,
        "method": label,
        "raw_method": rec.get("method"),
        "task_id": rec.get("task_id"),
        "passed": bool(passed) if passed is not None else None,
        "out_tok": rec.get("output_tokens"),
        "joules": rec.get("total_joules"),
        "joules_clean": rec.get("total_joules_clean"),
        "n_llm_calls": n,
        "eval_error": rec.get("eval_error"),
        "valid": bool(valid),
    }

def main():
    jobs = []
    for model, dirs in DIRS.items():
        for d in dirs:
            paths = _collect_episode_paths((Path(log_dir(d)),))
            jobs += [(model, str(p)) for p in paths]
    print(f"collecting {len(jobs)} episodes...", file=sys.stderr)
    out = []
    with ProcessPoolExecutor(max_workers=16) as ex:
        for r in ex.map(one, jobs, chunksize=8):
            if r:
                out.append(r)
    outp = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} records -> {outp}", file=sys.stderr)

if __name__ == "__main__":
    main()
