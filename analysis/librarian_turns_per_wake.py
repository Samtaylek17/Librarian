from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import (
    MODEL_DISPLAY, MODEL_ORDER, ROOT,
)

LIB_AGENTS = [
    "HyperAgentLibrarianMASRunner",
    "BOADLibrarianMASRunner",
]
TURNS_CSV = os.path.join(ROOT, "data", "turns.csv")
OUT_FIG = os.path.join(ROOT, "figures", "raw", "librarian_turns_per_wake.pdf")
OUT_FIG_PNG = os.path.join(ROOT, "figures", "raw", "librarian_turns_per_wake.png")
MIN_EPISODES = 10

MODEL_STYLE = {
    "Qwen/Qwen3.6-35B-A3B-FP8": ("-",  "o", "35B-A3B"),
    "Qwen/Qwen3.6-27B-FP8":     ("--", "^", "27B"),
}
SERIES_COLOR = {
    ("HyperAgentLibrarianMASRunner", "Qwen/Qwen3.6-35B-A3B-FP8"): "#2ca02c",
    ("HyperAgentLibrarianMASRunner", "Qwen/Qwen3.6-27B-FP8"):     "#1f77b4",
    ("BOADLibrarianMASRunner",       "Qwen/Qwen3.6-27B-FP8"):     "#d62728",
    ("BOADLibrarianMASRunner",       "Qwen/Qwen3.6-35B-A3B-FP8"): "#a8a8a8",
}
HOST_LABEL = {
    "HyperAgentLibrarianMASRunner": "HyperAgent",
    "BOADLibrarianMASRunner":       "BOAD",
}


def _is_lib(role: str) -> bool:
    return role == "navigator" or role.startswith("librarian")


def _wakes_per_episode() -> dict[tuple, list[int]]:
    per_ep: dict[tuple, list[tuple[int, str]]] = defaultdict(list)
    with open(TURNS_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r["agent_class"] not in LIB_AGENTS:
                continue
            key = (r["model"], r["agent_class"], r["task_id"], r["seed"])
            per_ep[key].append((int(r["turn_idx"]), r["agent_id"]))
    out: dict[tuple, list[int]] = {}
    for key, turns in per_ep.items():
        turns.sort()
        wakes: list[int] = []
        run = 0
        for _, role in turns:
            if _is_lib(role):
                run += 1
            else:
                if run:
                    wakes.append(run)
                    run = 0
        if run:
            wakes.append(run)
        out[key] = wakes
    return out


def main() -> None:
    wakes_by_ep = _wakes_per_episode()

    series: dict[tuple, defaultdict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (model, agent, _task, _seed), wakes in wakes_by_ep.items():
        for i, n in enumerate(wakes):
            series[(model, agent)][i + 1].append(n)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
    })
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for agent in LIB_AGENTS:
        host_label = HOST_LABEL[agent]
        for model in MODEL_ORDER:
            d = series.get((model, agent))
            if not d:
                continue
            xs, ys = [], []
            for pos in sorted(d.keys()):
                vals = d[pos]
                if len(vals) < MIN_EPISODES:
                    continue
                xs.append(pos)
                ys.append(sum(vals) / len(vals))
            if not xs:
                continue
            linestyle, marker, model_label = MODEL_STYLE[model]
            ax.plot(
                xs, ys,
                color=SERIES_COLOR[(agent, model)],
                linestyle=linestyle,
                marker=marker,
                markersize=4.5,
                linewidth=1.3,
                label=f"{host_label}, {model_label}",
            )

    ax.set_xlabel("Librarian invocation $k$")
    ax.set_ylabel("Mean LLM turns at invocation $k$")
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0.5)
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fontsize=7.5,
        handlelength=2.4,
        borderpad=0.4,
        labelspacing=0.3,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, bbox_inches="tight", pad_inches=0)
    fig.savefig(OUT_FIG_PNG, bbox_inches="tight", pad_inches=0, dpi=150)
    print(f"wrote {OUT_FIG}")
    print(f"wrote {OUT_FIG_PNG}")

    print()
    for model in MODEL_ORDER:
        print(f"\n[{MODEL_DISPLAY[model]}] mean turns at wake position (dropped if n<{MIN_EPISODES})")
        for agent in LIB_AGENTS:
            d = series.get((model, agent), {})
            cells = []
            for pos in sorted(d.keys()):
                vals = d[pos]
                if len(vals) < MIN_EPISODES:
                    continue
                cells.append(f"k={pos}: {sum(vals)/len(vals):.2f} (n={len(vals)})")
            print(f"  {agent:34s}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
