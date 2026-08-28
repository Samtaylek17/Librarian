from __future__ import annotations

import csv
import os
import statistics
from collections import defaultdict

import numpy as np

from _common import (
    AGENT_DISPLAY, AGENT_FAMILY, AGENT_ORDER, MODEL_DISPLAY, MODEL_ORDER, ROOT,
    ROW_LABEL, load_rows, mean,
)

BIN_WIDTH = 32_000
BIN_CAP   = 96_000
OUT_CSV_PASS = os.path.join(ROOT, "tables", "data", "difficulty_bins.csv")
OUT_CSV_ENERGY = os.path.join(ROOT, "tables", "data", "difficulty_energy.csv")


def _compute_bins(rows: list[dict]) -> tuple[dict, list[float], int]:
    vanilla = [r for r in rows if r["agent"] == "VanillaRunner"]
    by_task: dict[str, list[float]] = defaultdict(list)
    for r in vanilla:
        by_task[r["task_id"]].append(r["max_input_tokens"])
    task_diff = {t: statistics.mean(v) for t, v in by_task.items()}

    n_full = int(np.ceil(BIN_CAP / BIN_WIDTH))
    edges = [i * BIN_WIDTH for i in range(n_full + 1)] + [float("inf")]
    n_bins = n_full + 1

    task_bin: dict[str, int] = {}
    for t, d in task_diff.items():
        b = min(int(d // BIN_WIDTH), n_full)
        task_bin[t] = b
    return task_bin, edges, n_bins


def _bin_label(edges: list[float], i: int) -> str:
    lo = edges[i] / 1000.0
    hi = edges[i + 1]
    if hi == float("inf"):
        return f">{lo:.0f}K"
    return f"{lo:.0f}--{hi/1000:.0f}K"


def main() -> None:
    rows = load_rows()
    task_bin, edges, n_bins = _compute_bins(rows)

    agg: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: {"pass": [], "energy": []}
    )
    for r in rows:
        b = task_bin.get(r["task_id"])
        if b is None:
            continue
        key = (r["model"], r["agent"], b)
        agg[key]["pass"].append(r["passed"])
        agg[key]["energy"].append(r["total_joules_clean"])

    bin_size = [sum(1 for v in task_bin.values() if v == b) for b in range(n_bins)]
    print(f"bin width: {BIN_WIDTH/1000:.0f}K, n_bins: {n_bins}")
    print("bin sizes (tasks):", bin_size)

    def bin_range(i: int) -> str:
        return _bin_label(edges, i)

    def bin_n(i: int) -> str:
        return f"(n={bin_size[i]})"

    def _write_table(metric: str, scale: float, fmt: str, path: str,
                     bold_extreme: str = "max") -> None:
        fields = ["block", "model", "fam", "label"]
        for b in range(n_bins):
            fields += [f"b{b}", f"b{b}b"]

        out_rows: list[dict] = []
        for model in MODEL_ORDER:
            agents_here = [
                a for a in AGENT_ORDER
                if AGENT_FAMILY[a][0] in ("hyperagent", "boad")
            ]
            fam_order: list[str] = []
            fam_to_agents: dict[str, list[str]] = {}
            for a in agents_here:
                fam = AGENT_FAMILY[a][0]
                if fam not in fam_to_agents:
                    fam_order.append(fam)
                    fam_to_agents[fam] = []
                fam_to_agents[fam].append(a)

            for fam in fam_order:
                pair = fam_to_agents[fam]
                pair_vals: dict[str, list[float | None]] = {a: [] for a in pair}
                for a in pair:
                    for b in range(n_bins):
                        vals = [v for v in agg[(model, a, b)][metric] if v == v]
                        pair_vals[a].append(
                            (sum(vals) / len(vals)) * scale if vals else None
                        )
                winners_per_bin: list[set[str]] = []
                for b in range(n_bins):
                    col = [(a, pair_vals[a][b]) for a in pair if pair_vals[a][b] is not None]
                    if not col:
                        winners_per_bin.append(set())
                        continue
                    target = (max if bold_extreme == "max" else min)(v for _, v in col)
                    winners_per_bin.append({a for a, v in col if v == target})
                for a in pair:
                    row = {
                        "block": f"{MODEL_DISPLAY[model]}/{fam}",
                        "model": MODEL_DISPLAY[model],
                        "fam": fam,
                        "label": ROW_LABEL[a],
                    }
                    for b in range(n_bins):
                        v = pair_vals[a][b]
                        if v is None:
                            row[f"b{b}"], row[f"b{b}b"] = "--", 0
                        else:
                            row[f"b{b}"] = format(v, fmt)
                            row[f"b{b}b"] = 1 if a in winners_per_bin[b] else 0
                    out_rows.append(row)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)
        print(f"wrote {path}")

    _write_table("pass", 100.0, ".1f", OUT_CSV_PASS, bold_extreme="max")
    _write_table(
        "energy", 1.0 / 1000.0, ".2f", OUT_CSV_ENERGY, bold_extreme="min"
    )

    print()
    print("Bin edges (K tokens):", [f"{e/1000:.1f}" for e in edges])
    for model in MODEL_ORDER:
        print(f"\n[{MODEL_DISPLAY[model]}] pass% by bin")
        for agent in AGENT_ORDER:
            row = []
            for b in range(n_bins):
                vals = agg[(model, agent, b)]["pass"]
                row.append(f"{100*mean(vals):5.1f}" if vals else "  -- ")
            print(f"  {AGENT_DISPLAY[agent]:16s} " + " ".join(row))


if __name__ == "__main__":
    main()
