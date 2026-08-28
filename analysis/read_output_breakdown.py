from __future__ import annotations

import csv
import os
from collections import defaultdict

from _common import MODEL_DISPLAY, MODEL_ORDER, ROOT

TURNS = os.path.join(ROOT, "data", "turns.csv")
OUT_CSV = os.path.join(ROOT, "tables", "data", "read_output_breakdown.csv")

METHODS = [
    ("HyperAgent", "HyperAgentMASRunner"),
    ("BOAD", "BOADMASRunner"),
]


def main() -> None:
    tot: dict[tuple, float] = defaultdict(float)
    read: dict[tuple, float] = defaultdict(float)
    dup: dict[tuple, float] = defaultdict(float)
    episodes: dict[tuple, set] = defaultdict(set)
    classes = {ac for _, ac in METHODS}

    with open(TURNS, newline="") as f:
        for r in csv.DictReader(f):
            ac = r["agent_class"]
            if ac not in classes:
                continue
            m = r["model"]
            o = float(r["output_tokens"])
            tot[(m, ac)] += o
            if r["is_file_read"] == "1":
                read[(m, ac)] += o
            if r["is_duplicate"] == "1":
                dup[(m, ac)] += o
            episodes[(m, ac)].add((r["task_id"], r["seed"]))

    fields = ["block", "label", "out", "read", "dup"]
    out_rows: list[dict] = []
    for model in MODEL_ORDER:
        for disp, ac in METHODS:
            k = (model, ac)
            n = len(episodes[k]) or 1
            out_rows.append({
                "block": MODEL_DISPLAY[model],
                "label": disp,
                "out": f"{tot[k]/n/1e3:.2f}",
                "read": f"{read[k]/n/1e3:.2f}",
                "dup": f"{dup[k]/n/1e3:.2f}",
            })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT_CSV}")

    print()
    for model in MODEL_ORDER:
        print(f"[{MODEL_DISPLAY[model]}]")
        for disp, ac in METHODS:
            k = (model, ac)
            n = len(episodes[k]) or 1
            print(f"  {disp:12s} out={tot[k]/n/1e3:7.2f}K  read={read[k]/n/1e3:6.2f}K  dup={dup[k]/n/1e3:6.2f}K  (n={n})")


if __name__ == "__main__":
    main()
