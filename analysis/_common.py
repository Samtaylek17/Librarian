from __future__ import annotations

import csv
import os
from typing import Iterable

ROOT = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.dirname(ROOT)
CSV_PATH = os.path.join(ROOT, "data", "episodes.csv")
TURNS_PATH = os.path.join(ROOT, "data", "turns.csv")
TOKENEFF_CSV_PATH = os.path.join(ROOT, "data", "episodes_tokenefficiency.csv")
TOKENEFF_TURNS_PATH = os.path.join(ROOT, "data", "turns_tokenefficiency.csv")


# ── Raw episode logs ────────────────────────────────────────────────
# Scripts that read raw episode JSON (the significance / audit
# pipelines) resolve a matrix title through this search path. Matrices run in
# this checkout land in ``<repo>/logs``; archived main-results matrices often
# live on a separate volume. Point ``EPISODE_LOG_PATH`` at them, e.g.
#   export EPISODE_LOG_PATH=/data/librarian_logs:$PWD/logs
EPISODE_LOG_PATH = [
    p for p in os.environ.get("EPISODE_LOG_PATH", "").split(os.pathsep) if p
] or [os.path.join(REPO_ROOT, "logs")]


def log_dir(title: str) -> str:
    """Resolve a matrix title (a ``logs/<title>`` dir name) on EPISODE_LOG_PATH."""
    for root in EPISODE_LOG_PATH:
        cand = os.path.join(root, title)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        f"log dir {title!r} not found on EPISODE_LOG_PATH="
        f"{os.pathsep.join(EPISODE_LOG_PATH)}. Set EPISODE_LOG_PATH to the "
        f"volume(s) holding your matrix logs."
    )


TOKENEFF_METHOD_TO_AGENT = {
    "hyperagent_caveman":           "HyperAgentCavemanRunner",
    "hyperagent_librarian_caveman": "HyperAgentCavemanLibRunner",
    "hyperagent_lastn":             "HyperAgentLastNRunner",
    "boad_caveman":                 "BOADCavemanRunner",
    "boad_librarian_caveman":       "BOADCavemanLibRunner",
    "boad_lastn":                   "BOADLastNRunner",
}

NUM_COLS = (
    "passed", "score",
    "input_tokens", "output_tokens", "cached_tokens", "cache_h_tokens",
    "max_input_tokens", "n_llm_calls",
    "total_gpu_duration_seconds", "total_joules_clean",
    "agent_duration_seconds", "total_duration_seconds",
)

MODEL_DISPLAY = {
    "Qwen/Qwen3.6-27B-FP8": "Qwen3.6-27B",
    "Qwen/Qwen3.6-35B-A3B-FP8": "Qwen3.6-35B-A3B",
}
MODEL_SHORT = {
    "Qwen/Qwen3.6-27B-FP8": "27B",
    "Qwen/Qwen3.6-35B-A3B-FP8": "35B-A3B",
}
AGENT_DISPLAY = {
    "HyperAgentMASRunner": "HyperAgent",
    "HyperAgentLibrarianMASRunner": "HyperAgent+Lib",
    "HyperAgentLastNRunner": "HyperAgent+LastN",
    "HyperAgentCavemanRunner": "HyperAgent+Caveman",
    "HyperAgentCavemanLibRunner": "HyperAgent+Caveman+Lib",
    "BOADMASRunner": "BOAD",
    "BOADLibrarianMASRunner": "BOAD+Lib",
    "BOADLastNRunner": "BOAD+LastN",
    "BOADCavemanRunner": "BOAD+Caveman",
    "BOADCavemanLibRunner": "BOAD+Caveman+Lib",
}
AGENT_ORDER = [
    "HyperAgentMASRunner",
    "HyperAgentLibrarianMASRunner",
    "HyperAgentLastNRunner",
    "HyperAgentCavemanRunner",
    "HyperAgentCavemanLibRunner",
    "BOADMASRunner",
    "BOADLibrarianMASRunner",
    "BOADLastNRunner",
    "BOADCavemanRunner",
    "BOADCavemanLibRunner",
]
MODEL_ORDER = ["Qwen/Qwen3.6-35B-A3B-FP8", "Qwen/Qwen3.6-27B-FP8"]

AGENT_FAMILY = {
    "HyperAgentMASRunner":            ("hyperagent", False),
    "HyperAgentLibrarianMASRunner":   ("hyperagent", True),
    "HyperAgentLastNRunner":          ("hyperagent", False),
    "HyperAgentCavemanRunner":        ("hyperagent", False),
    "HyperAgentCavemanLibRunner":     ("hyperagent", True),
    "BOADMASRunner":                  ("boad",      False),
    "BOADLibrarianMASRunner":         ("boad",      True),
    "BOADLastNRunner":                ("boad",      False),
    "BOADCavemanRunner":              ("boad",      False),
    "BOADCavemanLibRunner":           ("boad",      True),
}
FAMILY_COLOR = {
    "hyperagent": "#2ca02c",
    "boad":      "#d62728",
}

METHOD_ORDER = ["hyperagent", "boad"]
METHOD_DISPLAY = {"hyperagent": "HyperAgent", "boad": "BOAD"}
VARIANT_ORDER = ["base", "librarian", "lastn", "caveman", "caveman_lib"]
VARIANT_DISPLAY = {
    "base":        "Base",
    "librarian":   "$+$ Librarian",
    "lastn":       "$+$ LastNObservation",
    "caveman":     "$+$ Caveman",
    "caveman_lib": "$+$ Caveman $+$ Librarian",
}
METHOD_VARIANT_AGENT = {
    ("hyperagent", "base"):        "HyperAgentMASRunner",
    ("hyperagent", "librarian"):   "HyperAgentLibrarianMASRunner",
    ("hyperagent", "lastn"):       "HyperAgentLastNRunner",
    ("hyperagent", "caveman"):     "HyperAgentCavemanRunner",
    ("hyperagent", "caveman_lib"): "HyperAgentCavemanLibRunner",
    ("boad",       "base"):        "BOADMASRunner",
    ("boad",       "librarian"):   "BOADLibrarianMASRunner",
    ("boad",       "lastn"):       "BOADLastNRunner",
    ("boad",       "caveman"):     "BOADCavemanRunner",
    ("boad",       "caveman_lib"): "BOADCavemanLibRunner",
}


ROW_LABEL = {
    "HyperAgentMASRunner":          "HyperAgent",
    "HyperAgentLibrarianMASRunner": "\\hspace{1ex}$+$ Librarian",
    "HyperAgentLastNRunner":        "\\hspace{1ex}$+$ LastNObservation",
    "HyperAgentCavemanRunner":      "\\hspace{1ex}$+$ Caveman",
    "HyperAgentCavemanLibRunner":   "\\hspace{1ex}$+$ Caveman $+$ Librarian",
    "BOADMASRunner":                "BOAD",
    "BOADLibrarianMASRunner":       "\\hspace{1ex}$+$ Librarian",
    "BOADLastNRunner":              "\\hspace{1ex}$+$ LastNObservation",
    "BOADCavemanRunner":            "\\hspace{1ex}$+$ Caveman",
    "BOADCavemanLibRunner":         "\\hspace{1ex}$+$ Caveman $+$ Librarian",
}


def variant_row_label(method: str, variant: str) -> str:
    if variant == "base":
        return METHOD_DISPLAY[method]
    return f"\\hspace{{1ex}}{VARIANT_DISPLAY[variant]}"


def agent_style(agent: str) -> dict:
    family, is_lib = AGENT_FAMILY[agent]
    return {
        "color": FAMILY_COLOR[family],
        "linestyle": "--" if is_lib else "-",
        "marker": "s" if is_lib else "o",
        "markersize": 5,
        "linewidth": 1.6,
    }


def _coerce(row: dict) -> dict:
    for c in NUM_COLS:
        v = row.get(c, "")
        if v in ("", None):
            row[c] = float("nan")
        elif isinstance(v, str) and v.strip().lower() in ("true", "false"):
            row[c] = 1.0 if v.strip().lower() == "true" else 0.0
        else:
            row[c] = float(v)
    return row


def load_rows(path: str = CSV_PATH, include_tokeneff: bool = True) -> list[dict]:
    with open(path, newline="") as f:
        rows = [_coerce(r) for r in csv.DictReader(f)]
    if include_tokeneff and os.path.exists(TOKENEFF_CSV_PATH):
        with open(TOKENEFF_CSV_PATH, newline="") as f:
            for r in csv.DictReader(f):
                agent = TOKENEFF_METHOD_TO_AGENT.get(r.get("method", ""))
                if agent is None:
                    continue
                r["agent"] = agent
                rows.append(_coerce(r))
    return rows


def iter_turn_rows(include_tokeneff: bool = True):
    with open(TURNS_PATH, newline="") as f:
        yield from csv.DictReader(f)
    if include_tokeneff and os.path.exists(TOKENEFF_TURNS_PATH):
        with open(TOKENEFF_TURNS_PATH, newline="") as f:
            for r in csv.DictReader(f):
                agent = TOKENEFF_METHOD_TO_AGENT.get(r.get("method", ""))
                if agent is None:
                    continue
                r["agent_class"] = agent
                yield r


def group_by(rows: Iterable[dict], keys: tuple[str, ...]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        k = tuple(r[c] for c in keys)
        out.setdefault(k, []).append(r)
    return out


def mean(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def total(xs: Iterable[float]) -> float:
    return sum(x for x in xs if x == x)


def group_header_row(n_cols: int, label: str, color: str = "gray!12") -> str:
    return (
        f"\\rowcolor{{{color}}}[0pt][0pt]\n"
        f"\\multicolumn{{{n_cols}}}{{@{{}}l@{{}}}}{{\\textbf{{{label}}}}} \\\\"
    )


def dup_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "dup_blue", [(0.0, "#ffffff"), (0.2, "#c0e4ff"), (1.0, "#1971c2")],
    )


def heatmap_text_color(cmap, value: float, vmax: float) -> str:
    frac = 0.0 if vmax <= 0 else max(0.0, min(1.0, value / vmax))
    r, g, b, _ = cmap(frac)
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "black"
