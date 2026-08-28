from __future__ import annotations

import csv
import os

from _common import (
    METHOD_DISPLAY, METHOD_ORDER, METHOD_VARIANT_AGENT,
    MODEL_DISPLAY, MODEL_ORDER, ROOT,
    VARIANT_DISPLAY, VARIANT_ORDER,
    group_by, load_rows, mean, total, variant_row_label,
)

OUT_CSV = os.path.join(ROOT, "tables", "data", "main_results.csv")

COLUMNS = [
    ("pass_pct",  "pass",   "Pass",    "(\\%)", ".1f", "max"),
    ("out_tok_K", "out",    "Out tok", "(K)",   ".1f", "min"),
    ("energy_kJ", "energy", "Energy",  "(kJ)",  ".2f", "min"),
]


def _compute_stats(rows: list[dict]) -> dict:
    groups = group_by(rows, ("model", "agent"))
    stats = {}
    for (model, agent), recs in groups.items():
        stats[(model, agent)] = {
            "pass_pct":     100.0 * mean(r["passed"] for r in recs),
            "uncached_tok_M": mean(r["input_tokens"] - r["cached_tokens"] for r in recs) / 1e6,
            "cached_tok_M": mean(r["cached_tokens"] for r in recs) / 1e6,
            "out_tok_K":    mean(r["output_tokens"] for r in recs) / 1e3,
            "energy_kJ":    mean(r["total_joules_clean"] for r in recs) / 1e3,
            "gpu_time_min": mean(r["total_gpu_duration_seconds"] for r in recs) / 60.0,
        }
    return stats


def _block_best(block_stats: list[dict | None], key: str, direction: str | None):
    if direction is None:
        return None
    vals = [s[key] for s in block_stats if s is not None]
    if not vals:
        return None
    return max(vals) if direction == "max" else min(vals)


def main() -> None:
    rows = load_rows()
    stats = _compute_stats(rows)

    fields = ["block", "model", "meth", "label"]
    for _, short, *_ in COLUMNS:
        fields += [short, short + "b"]

    out_rows: list[dict] = []
    for model in MODEL_ORDER:
        for method in METHOD_ORDER:
            block = [
                stats.get((model, METHOD_VARIANT_AGENT[(method, v)]))
                if METHOD_VARIANT_AGENT[(method, v)] else None
                for v in VARIANT_ORDER
            ]
            best = {
                key: _block_best(block, key, direction)
                for key, _, _, _, _, direction in COLUMNS
            }
            for vi, variant in enumerate(VARIANT_ORDER):
                s = block[vi]
                row = {
                    "block": f"{MODEL_DISPLAY[model]}/{method}",
                    "model": MODEL_DISPLAY[model],
                    "meth": method,
                    "label": variant_row_label(method, variant),
                }
                for key, short, _, _, fmt, _ in COLUMNS:
                    if s is None:
                        row[short], row[short + "b"] = "--", 0
                    else:
                        is_best = best[key] is not None and abs(s[key] - best[key]) < 1e-9
                        row[short] = format(s[key], fmt)
                        row[short + "b"] = 1 if is_best else 0
                out_rows.append(row)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT_CSV}")

    print()
    hdr = (
        f"{'Model':18s} {'Method':12s} {'Variant':22s} "
        + " ".join(f"{c[2]:>9s}" for c in COLUMNS)
    )
    print(hdr)
    print("-" * len(hdr))
    for model in MODEL_ORDER:
        for method in METHOD_ORDER:
            for variant in VARIANT_ORDER:
                agent = METHOD_VARIANT_AGENT[(method, variant)]
                s = stats.get((model, agent)) if agent else None
                cells = []
                for key, _, _, _, fmt, _ in COLUMNS:
                    cells.append("--" if s is None else format(s[key], fmt))
                print(
                    f"{MODEL_DISPLAY[model]:18s} {METHOD_DISPLAY[method]:12s} "
                    f"{VARIANT_DISPLAY[variant]:22s} "
                    + " ".join(f"{c:>9s}" for c in cells)
                )


if __name__ == "__main__":
    main()
