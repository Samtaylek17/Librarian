from __future__ import annotations

import csv
import os
import random
from collections import defaultdict

from _common import (
    METHOD_DISPLAY, METHOD_ORDER, MODEL_DISPLAY, MODEL_ORDER, ROOT,
    load_rows, mean,
)

OUT_CSV = os.path.join(ROOT, "tables", "data", "retrieval_comparison.csv")
COMMON_SAMPLE_SIZE = 100
RANDOM_SEED = 0

def _is_hyperagent_base(a, d):    return a == "HyperAgentMASRunner"
def _is_hyperagent_sparse(a, d):  return a == "HyperAgentLibrarianRAGMASRunner" and "bm25" in d
def _is_hyperagent_dense(a, d):   return a == "HyperAgentLibrarianRAGMASRunner" and "sfr_code" in d
def _is_hyperagent_lib(a, d):     return a == "HyperAgentLibrarianMASRunner"

def _is_boad_base(a, d):          return a == "BOADMASRunner"
def _is_boad_sparse(a, d):        return a == "BOADLibrarianRAGMASRunner"   and "bm25" in d
def _is_boad_dense(a, d):         return a == "BOADLibrarianRAGMASRunner"   and "sfr_code" in d
def _is_boad_lib(a, d):           return a == "BOADLibrarianMASRunner"

CELLS: dict[tuple[str, str], callable] = {
    ("hyperagent", "base"):      _is_hyperagent_base,
    ("hyperagent", "sparse"):    _is_hyperagent_sparse,
    ("hyperagent", "dense"):     _is_hyperagent_dense,
    ("hyperagent", "librarian"): _is_hyperagent_lib,
    ("boad",       "base"):      _is_boad_base,
    ("boad",       "sparse"):    _is_boad_sparse,
    ("boad",       "dense"):     _is_boad_dense,
    ("boad",       "librarian"): _is_boad_lib,
}
VARIANT_ORDER = ["base", "sparse", "dense", "librarian"]
VARIANT_LABEL = {
    "base":      None,
    "sparse":    "\\hspace{1ex}$+$ Sparse retrieval",
    "dense":     "\\hspace{1ex}$+$ Dense retrieval",
    "librarian": "\\hspace{1ex}$+$ Librarian",
}


def _bin_episodes(rows: list[dict]) -> dict[tuple[str, str, str, str], list[dict]]:
    out: dict = defaultdict(list)
    for r in rows:
        a = r.get("agent", "") or ""
        d = r.get("agent_dir", "") or ""
        for (family, variant), match in CELLS.items():
            if match(a, d):
                out[(r["model"], family, variant, r["task_id"])].append(r)
                break
    return out


def _common_tasks(binned, model: str) -> set[str]:
    sets: list[set[str]] = []
    for (family, variant) in CELLS:
        s = {
            t for (m, f, v, t) in binned
            if m == model and f == family and v == variant
        }
        sets.append(s)
    return set.intersection(*sets) if sets else set()


def _sampled_tasks(common: set[str], n: int, seed: int) -> set[str]:
    pool = sorted(common)
    if len(pool) <= n:
        return set(pool)
    rng = random.Random(seed)
    return set(rng.sample(pool, n))


def _stats_on(binned, model, family, variant, tasks):
    passed, in_tok, out_tok, energy = [], [], [], []
    for t in tasks:
        recs = binned.get((model, family, variant, t), [])
        if not recs:
            continue
        r = recs[0]
        try:
            passed.append(float(r["passed"]))
            in_tok.append(float(r["input_tokens"]))
            out_tok.append(float(r["output_tokens"]))
            energy.append(float(r["total_joules_clean"]))
        except (ValueError, KeyError):
            continue
    return {
        "pass_pct":   100.0 * mean(passed)            if passed  else float("nan"),
        "in_tok_M":   mean(in_tok)   / 1e6            if in_tok  else float("nan"),
        "out_tok_K":  mean(out_tok)  / 1e3            if out_tok else float("nan"),
        "energy_kJ":  mean(energy)   / 1e3            if energy  else float("nan"),
        "n":          len(passed),
    }


COLUMNS = [
    ("pass_pct",  "Pass",    "(\\%)", ".1f", "max"),
    ("out_tok_K", "Out tok", "(K)",   ".1f", "min"),
    ("energy_kJ", "Energy",  "(kJ)",  ".2f", "min"),
]


COL_CSV = {
    "pass_pct":  "pass",
    "out_tok_K": "out",
    "energy_kJ": "energy",
}


def main() -> None:
    rows = load_rows()
    binned = _bin_episodes(rows)

    per_model_common: dict[str, set[str]] = {}
    per_model_sample: dict[str, set[str]] = {}
    stats: dict[tuple[str, str, str], dict] = {}
    for model in MODEL_ORDER:
        common = _common_tasks(binned, model)
        sample = _sampled_tasks(common, COMMON_SAMPLE_SIZE, RANDOM_SEED)
        per_model_common[model] = common
        per_model_sample[model] = sample
        for family, variant in CELLS:
            stats[(model, family, variant)] = _stats_on(
                binned, model, family, variant, sample
            )

    fields = ["block", "label"]
    for key, *_ in COLUMNS:
        short = COL_CSV[key]
        fields += [short, short + "b"]

    out_rows: list[dict] = []
    for model in MODEL_ORDER:
        for family in METHOD_ORDER:
            family_stats = [
                stats[(model, family, v)] for v in VARIANT_ORDER
            ]
            best = {}
            for key, _, _, _, direction in COLUMNS:
                vals = [s[key] for s in family_stats if s[key] == s[key]]
                if not vals:
                    best[key] = None
                else:
                    best[key] = max(vals) if direction == "max" else min(vals)
            for variant in VARIANT_ORDER:
                s = stats[(model, family, variant)]
                if variant == "base":
                    label = METHOD_DISPLAY[family]
                else:
                    label = VARIANT_LABEL[variant]
                row = {
                    "block": f"{MODEL_DISPLAY[model]}/{family}",
                    "label": label,
                }
                for key, _, _, fmt, _ in COLUMNS:
                    short = COL_CSV[key]
                    v = s[key]
                    if v != v:
                        row[short], row[short + "b"] = "--", 0
                    else:
                        is_best = (best[key] is not None
                                   and abs(v - best[key]) < 1e-9)
                        row[short] = format(v, fmt)
                        row[short + "b"] = 1 if is_best else 0
                out_rows.append(row)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT_CSV}")

    print()
    for model in MODEL_ORDER:
        common = per_model_common[model]
        sample = per_model_sample[model]
        print(f"=== {MODEL_DISPLAY[model]}  n_common={len(common)}  n_sampled={len(sample)} ===")
        for family in METHOD_ORDER:
            for variant in VARIANT_ORDER:
                s = stats[(model, family, variant)]
                if variant == "base":
                    lab = METHOD_DISPLAY[family]
                else:
                    lab = f"  + {variant}"
                print(
                    f"  {lab:20s} n={s['n']:3d} pass={s['pass_pct']:5.1f} "
                    f"in={s['in_tok_M']:5.2f}M out={s['out_tok_K']:6.2f}K "
                    f"E={s['energy_kJ']:6.2f}kJ"
                )


if __name__ == "__main__":
    main()
