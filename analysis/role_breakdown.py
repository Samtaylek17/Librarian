from __future__ import annotations

import csv
import os
import re
from collections import defaultdict

from _common import MODEL_DISPLAY, MODEL_ORDER, ROOT, iter_turn_rows

OUT_DIR = os.path.join(ROOT, "tables")
DATA_DIR = os.path.join(OUT_DIR, "data")

FAMILIES = [
    (
        "HyperAgent",
        "HyperAgentMASRunner",
        "HyperAgentLibrarianMASRunner",
        [
            ("planner",                              "planner",   "planner"),
            ("navigator $\\rightarrow$ librarian",   "navigator", "librarian"),
            ("editor",                               "editor",    "editor"),
            ("executor",                             "executor",  "executor"),
        ],
        "hyperagent",
    ),
    (
        "BOAD",
        "BOADMASRunner",
        "BOADLibrarianMASRunner",
        [
            ("orchestrator",                                     "orchestrator",   "orchestrator"),
            ("issue analyzer",                                   "issue_analyzer", "issue_analyzer"),
            ("code\\_navigator $\\rightarrow$ librarian",        "code_navigator", "librarian"),
        ],
        "boad",
    ),
]

MODEL_SHORT = {
    "Qwen/Qwen3.6-35B-A3B-FP8": "A3B",
    "Qwen/Qwen3.6-27B-FP8":     "27B",
}

FIELDS = ["block", "model", "role", "bout", "boutb", "be", "beb",
          "lout", "loutb", "le", "leb"]


def _role_of(agent_id: str) -> str:
    return re.sub(r"_\d+$", "", agent_id)


def _load_per_episode() -> dict:
    per_ep: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for r in iter_turn_rows():
        role = _role_of(r["agent_id"])
        k = (r["model"], r["agent_class"], r["task_id"], r["seed"], role)
        try:
            per_ep[k][0] += float(r["input_tokens"])
            per_ep[k][1] += float(r["output_tokens"])
            per_ep[k][2] += float(r["joules"])
            per_ep[k][3] += float(r["duration_seconds"])
        except ValueError:
            continue
    agg: dict[tuple, list[tuple]] = defaultdict(list)
    for (m, a, t, s, role), v in per_ep.items():
        agg[(m, a, role)].append(tuple(v))
    return agg


def _means(vals: list[tuple]) -> tuple[float, float, float, float]:
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    s_in = sum(v[0] for v in vals) / n
    s_out = sum(v[1] for v in vals) / n
    s_j   = sum(v[2] for v in vals) / n
    s_d   = sum(v[3] for v in vals) / n
    return s_in, s_out, s_j, s_d


def _fmt(v: float, fmt: str) -> str:
    return "--" if v != v else format(v, fmt)


def _pair_cells(b: float, l: float, fmt: str, bold_smaller: bool):
    sb = _fmt(b, fmt)
    sl = _fmt(l, fmt)
    bb = lb = 0
    if bold_smaller and sb != "--" and sl != "--" and abs(b - l) >= 1e-9:
        if b < l:
            bb = 1
        else:
            lb = 1
    return (sb, bb), (sl, lb)


def _write_csv(path: str, out_rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {path}")


def _build_rows(base_cls: str, lib_cls: str, rows: list, agg: dict,
                bold_smaller: bool) -> list[dict]:
    out_rows: list[dict] = []
    n_rows = len(rows)
    for model in MODEL_ORDER:
        for ri, (row_label, base_role, lib_role) in enumerate(rows):
            b_in, b_out, b_j, b_d = _means(agg.get((model, base_cls, base_role), []))
            l_in, l_out, l_j, l_d = _means(agg.get((model, lib_cls,  lib_role),  []))
            (b_ot, b_otb), (l_ot, l_otb) = _pair_cells(b_out/1e3, l_out/1e3, ".2f", bold_smaller)
            (b_e,  b_eb),  (l_e,  l_eb)  = _pair_cells(b_j/1e3,   l_j/1e3,   ".2f", bold_smaller)
            model_cell = (
                f"\\multirow{{{n_rows}}}{{*}}{{{MODEL_SHORT[model]}}}" if ri == 0
                else ""
            )
            out_rows.append({
                "block": MODEL_SHORT[model],
                "model": model_cell,
                "role": row_label,
                "bout": b_ot, "boutb": b_otb,
                "be": b_e, "beb": b_eb,
                "lout": l_ot, "loutb": l_otb,
                "le": l_e, "leb": l_eb,
            })
    return out_rows


def _build_compact_rows(agg: dict) -> list[dict]:
    variants: list[tuple[str, str, str, str, str]] = []
    for label, base_cls, lib_cls, rows, _slug in FAMILIES:
        nav_row = next(r for r in rows if "rightarrow" in r[0])
        _, base_role, lib_role = nav_row
        prefix = "HyperAgent" if "HyperAgent" in base_cls else "BOAD"
        variants.append((label, base_cls, lib_cls, base_role, lib_role))
        variants.append((
            f"{label} $+$ Caveman",
            f"{prefix}CavemanRunner", f"{prefix}CavemanLibRunner",
            base_role, lib_role,
        ))

    out_rows: list[dict] = []
    for model in MODEL_ORDER:
        for vi, (method, base_cls, lib_cls, base_role, lib_role) in enumerate(variants):
            b = _means(agg.get((model, base_cls, base_role), []))
            l = _means(agg.get((model, lib_cls,  lib_role),  []))
            (b_ot, b_otb), (l_ot, l_otb) = _pair_cells(b[1]/1e3, l[1]/1e3, ".2f", True)
            (b_e,  b_eb),  (l_e,  l_eb)  = _pair_cells(b[2]/1e3, l[2]/1e3, ".2f", True)
            model_cell = (
                f"\\multirow{{{len(variants)}}}{{*}}{{{MODEL_SHORT[model]}}}"
                if vi == 0 else ""
            )
            out_rows.append({
                "block": MODEL_SHORT[model],
                "model": model_cell,
                "role": method,
                "bout": b_ot, "boutb": b_otb,
                "be": b_e, "beb": b_eb,
                "lout": l_ot, "loutb": l_otb,
                "le": l_e, "leb": l_eb,
            })
    return out_rows


def main() -> None:
    agg = _load_per_episode()
    os.makedirs(DATA_DIR, exist_ok=True)

    _write_csv(
        os.path.join(DATA_DIR, "role_comparison_compact.csv"),
        _build_compact_rows(agg),
    )

    for label, base_cls, lib_cls, rows, slug in FAMILIES:
        _write_csv(
            os.path.join(DATA_DIR, f"role_comparison_{slug}.csv"),
            _build_rows(base_cls, lib_cls, rows, agg, bold_smaller=False),
        )

    print()
    for label, base_cls, lib_cls, rows, _slug in FAMILIES:
        print(f"--- {label} ---")
        hdr = (
            f"{'model':10s} {'role':28s} "
            f"{'b_in(M)':>9s} {'b_out(K)':>9s} {'b_E(kJ)':>9s} {'b_T(min)':>9s}   "
            f"{'l_in(M)':>9s} {'l_out(K)':>9s} {'l_E(kJ)':>9s} {'l_T(min)':>9s}"
        )
        print(hdr)
        for model in MODEL_ORDER:
            for row_label, base_role, lib_role in rows:
                b_in, b_out, b_j, b_d = _means(agg.get((model, base_cls, base_role), []))
                l_in, l_out, l_j, l_d = _means(agg.get((model, lib_cls,  lib_role),  []))
                rl = re.sub(r"\\\\.|\\rightarrow", "->", row_label).replace("$", "")
                rl = re.sub(r"\\_", "_", rl)
                print(
                    f"{MODEL_SHORT[model]:10s} {rl:28s} "
                    f"{b_in/1e6:9.2f} {b_out/1e3:9.2f} {b_j/1e3:9.2f} {b_d/60:9.2f}   "
                    f"{l_in/1e6:9.2f} {l_out/1e3:9.2f} {l_j/1e3:9.2f} {l_d/60:9.2f}"
                )
        print()


if __name__ == "__main__":
    main()
