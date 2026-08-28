from __future__ import annotations

import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import MODEL_ORDER, ROOT, load_rows
from librarian_wakes_by_bin import _compute_bins, _wakes_per_episode

RAW = os.path.join(ROOT, "figures", "raw")
MODEL_SLUG = {
    "Qwen/Qwen3.6-35B-A3B-FP8": "a3b",
    "Qwen/Qwen3.6-27B-FP8":     "27b",
}

HA_COLOR = "#2ca02c"
BOAD_COLOR = "#d62728"
SERIES = [
    ("HyperAgentLibrarianMASRunner",   HA_COLOR,   None,   "HyperAgent + Librarian"),
    ("HyperAgentCavemanLibRunner",     HA_COLOR,   "////", "HyperAgent + Caveman + Librarian"),
    ("BOADLibrarianMASRunner",         BOAD_COLOR, None,   "BOAD + Librarian"),
    ("BOADCavemanLibRunner",           BOAD_COLOR, "////", "BOAD + Caveman + Librarian"),
]


def _bin_labels(edges) -> list[str]:
    labels = []
    for i in range(len(edges) - 1):
        lo = edges[i] / 1000.0
        hi = edges[i + 1]
        labels.append(f">{lo:.0f}K" if hi == float("inf") else f"{lo:.0f}–{hi/1000:.0f}K")
    return labels


def main() -> None:
    rows = load_rows()
    task_bin, edges, n_bins = _compute_bins(rows)
    wakes_by_ep = _wakes_per_episode()

    agg: dict[tuple, list[int]] = defaultdict(list)
    for (model, agent, task_id, _seed), wakes in wakes_by_ep.items():
        b = task_bin.get(task_id)
        if b is not None:
            agg[(model, agent, b)].append(len(wakes))

    def mean_at(model, agent, b):
        vals = agg.get((model, agent, b), [])
        return sum(vals) / len(vals) if vals else np.nan

    ymax = max(
        (m for model in MODEL_ORDER for a, *_ in SERIES
         for b in range(n_bins) if (m := mean_at(model, a, b)) == m),
        default=1.0,
    ) * 1.08

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    x = np.arange(n_bins)
    width = 0.2
    offsets = [(j - (len(SERIES) - 1) / 2) * width for j in range(len(SERIES))]
    labels = _bin_labels(edges)

    for mi, model in enumerate(MODEL_ORDER):
        fig, ax = plt.subplots(figsize=(3.7, 3.0))
        for j, (agent, color, hatch, label) in enumerate(SERIES):
            heights = [mean_at(model, agent, b) for b in range(n_bins)]
            ax.bar(
                x + offsets[j], heights, width,
                color=color, hatch=hatch, edgecolor="white", linewidth=0.5,
                label=label,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Reference max-input tokens (K)")
        ax.set_ylabel("Mean Librarian invocations")
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        if mi == 0:
            ax.legend(
                loc="upper left", fontsize=7, frameon=True, framealpha=0.9,
                handlelength=1.3, labelspacing=0.3, borderpad=0.4,
            )

        fig.tight_layout()
        slug = MODEL_SLUG[model]
        os.makedirs(RAW, exist_ok=True)
        fig.savefig(os.path.join(RAW, f"librarian_wakes_by_bin_{slug}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(RAW, f"librarian_wakes_by_bin_{slug}.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"wrote librarian_wakes_by_bin_{slug}.pdf")


if __name__ == "__main__":
    main()
