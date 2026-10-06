from __future__ import annotations

import csv
import os
from collections import defaultdict

from _common import MODEL_DISPLAY, MODEL_ORDER, ROOT, load_rows, mean

OUT_CSV = os.path.join(ROOT, "tables", "data", "total_vs_energy.csv")

COLUMNS = [
    ("uncached_M", "uncached", "{:.2f}"),
    ("cached_M",   "cached",   "{:.2f}"),
    ("out_K",      "out",      "{:.1f}"),
    ("total_M",    "total",    "{:.2f}"),
    ("energy_kJ",  "energy",   "{:.2f}"),
]

METHODS = [
    ("Vanilla",    "VanillaRunner"),
    ("HyperAgent", "HyperAgentMASRunner"),
    ("BOAD",       "BOADMASRunner"),
]


def _compute(rows: list[dict]) -> dict[tuple, dict[str, float]]:
    agg = defaultdict(list)
    for r in rows:
        agg[(r["model"], r["agent"])].append(r)
    out = {}
    for k, recs in agg.items():
        out[k] = {
            "uncached_M": mean(r["input_tokens"] - r["cached_tokens"] for r in recs) / 1e6,
            "cached_M":   mean(r["cached_tokens"] for r in recs) / 1e6,
            "out_K":      mean(r["output_tokens"] for r in recs) / 1e3,
            "total_M":    mean(r["input_tokens"] + r["output_tokens"] for r in recs) / 1e6,
            "energy_kJ":  mean(r["total_joules_clean"] for r in recs) / 1e3,
        }
    return out


def main() -> None:
    rows = load_rows()
    stats = _compute(rows)

    fields = ["block", "meth"]
    for _, short, _ in COLUMNS:
        fields += [short, short + "b"]

    out_rows: list[dict] = []
    for model in MODEL_ORDER:
        backbone = [
            stats[(model, agent_cls)]
            for _, agent_cls in METHODS
            if (model, agent_cls) in stats
        ]
        if not backbone:
            continue
        best = {key: min(s[key] for s in backbone) for key, _, _ in COLUMNS}
        for label, agent_cls in METHODS:
            s = stats.get((model, agent_cls))
            if s is None:
                continue
            row = {"block": MODEL_DISPLAY[model], "meth": label}
            for key, short, fmt in COLUMNS:
                row[short] = fmt.format(s[key])
                row[short + "b"] = 1 if s[key] == best[key] else 0
            out_rows.append(row)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT_CSV}")

    print()
    print(
        f"{'model':25s} {'method':12s} {'uncach_M':>9s} "
        f"{'cach_M':>8s} {'out_K':>8s} {'total_M':>9s} {'E_kJ':>8s}"
    )
    for model in MODEL_ORDER:
        for label, agent_cls in METHODS:
            s = stats.get((model, agent_cls))
            if s is None:
                continue
            print(
                f"{MODEL_DISPLAY[model]:25s} {label:12s} "
                f"{s['uncached_M']:9.2f} {s['cached_M']:8.2f} "
                f"{s['out_K']:8.1f} {s['total_M']:9.2f} {s['energy_kJ']:8.2f}"
            )


if __name__ == "__main__":
    main()
