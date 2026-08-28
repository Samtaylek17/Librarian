from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ROOT, dup_cmap, heatmap_text_color

CMAP = dup_cmap()

CSV_PATH   = os.path.join(ROOT, "data", "episodes.csv")
JSON_PATH  = os.path.join(ROOT, "data", "cross_role_pair.jsonl")
TURNS_PATH = os.path.join(ROOT, "data", "turns.csv")

A3B = "Qwen/Qwen3.6-35B-A3B-FP8"
B27 = "Qwen/Qwen3.6-27B-FP8"

SYSTEMS = {
    "boad": {
        "title":  "BOAD",
        "agent":  "BOADMASRunner",
        "roles":  ["orchestrator", "issue_analyzer", "code_navigator"],
        "labels": ["orch",         "issue",          "cnav"],
        "out_pdf": os.path.join(ROOT, "figures", "raw", "dup_output_heatmap_boad.pdf"),
        "out_png": os.path.join(ROOT, "figures", "raw", "dup_output_heatmap_boad.png"),
    },
    "boad_lib": {
        "title":  "BOAD+Lib",
        "agent":  "BOADLibrarianMASRunner",
        "roles":  ["orchestrator", "issue_analyzer", "librarian"],
        "labels": ["orch",         "issue",          "lib"],
        "out_pdf": os.path.join(ROOT, "figures", "raw", "dup_output_heatmap_boad_lib.pdf"),
        "out_png": os.path.join(ROOT, "figures", "raw", "dup_output_heatmap_boad_lib.png"),
    },
    "ha": {
        "title":  "HyperAgent",
        "agent":  "HyperAgentMASRunner",
        "roles":  ["navigator", "executor", "editor"],
        "labels": ["nav",       "exec",     "editor"],
        "out_pdf": os.path.join(ROOT, "figures", "raw", "dup_output_heatmap_ha.pdf"),
        "out_png": os.path.join(ROOT, "figures", "raw", "dup_output_heatmap_ha.png"),
    },
    "ha_lib": {
        "title":  "HyperAgent+Lib",
        "agent":  "HyperAgentLibrarianMASRunner",
        "roles":  ["librarian", "executor", "editor"],
        "labels": ["lib",       "exec",     "editor"],
        "out_pdf": os.path.join(ROOT, "figures", "raw", "dup_output_heatmap_ha_lib.pdf"),
        "out_png": os.path.join(ROOT, "figures", "raw", "dup_output_heatmap_ha_lib.png"),
    },
}


def _load_cross_agent(agent: str, roles: list[str]) -> np.ndarray:
    sums   = defaultdict(float)
    counts = defaultdict(int)
    with open(JSON_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["runner"] != agent:
                continue
            counts[r["model"]] += 1
            for k, v in r.get("pair_out", {}).items():
                cur, prior = k.split("__")
                if cur in roles and prior in roles:
                    sums[(r["model"], cur, prior)] += float(v)
    means = {k: sums[k] / counts[k[0]] for k in sums}

    n = len(roles)
    mat = np.zeros((n, n))
    for i, prior in enumerate(roles):
        for j, cur in enumerate(roles):
            mat[i, j] = (
                means.get((A3B, cur, prior), 0.0)
                + means.get((B27, cur, prior), 0.0)
            )
    return mat


def _load_same_agent(agent: str, roles: list[str]) -> np.ndarray:
    sums   = defaultdict(float)
    counts = defaultdict(int)
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            if r["agent"] != agent:
                continue
            try:
                per = json.loads(r["per_role_json"])
            except (ValueError, KeyError):
                continue
            counts[r["model"]] += 1
            for role in roles:
                v = per.get(role, {}).get("any_intra_out", 0)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = 0.0
                sums[(r["model"], role)] += v
    means = {k: sums[k] / counts[k[0]] for k in sums}
    return np.array([
        means.get((A3B, role), 0.0) + means.get((B27, role), 0.0)
        for role in roles
    ])


def _read_out_denominators(agents: list[str]) -> dict[str, float]:
    want = set(agents)
    sums:     dict[tuple[str, str], float] = defaultdict(float)
    episodes: dict[tuple[str, str], set]   = defaultdict(set)
    with open(TURNS_PATH, newline="") as f:
        for r in csv.DictReader(f):
            ac = r["agent_class"]
            if ac not in want:
                continue
            key = (ac, r["model"])
            episodes[key].add((r["task_id"], r["seed"]))
            if r["is_file_read"] == "1":
                sums[key] += float(r["output_tokens"])
    return {
        ac: sum(
            sums[(ac, m)] / (len(episodes[(ac, m)]) or 1) for m in (A3B, B27)
        )
        for ac in want
    }


def _annotate_cells(ax, data: np.ndarray, vmax: float) -> None:
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            color = heatmap_text_color(CMAP, v, vmax)
            ax.text(j, i, f"{100 * v:.1f}", ha="center", va="center", color=color)


def _draw_panel(title: str, labels: list[str],
                cross: np.ndarray, same: np.ndarray, denom: float,
                out_pdf: str, out_png: str) -> None:
    n = len(labels)
    cross = cross / denom
    same = same / denom
    same_col = same.reshape(-1, 1)
    vmax = float(max(cross.max(), same.max())) or 1.0

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 14})
    fig = plt.figure(figsize=(6.0, 3.0))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1, n], wspace=0.55,
        left=0.14, right=0.78, top=1.0, bottom=0.0,
    )
    ax_same  = fig.add_subplot(gs[0, 0])
    ax_cross = fig.add_subplot(gs[0, 1])

    ax_same.imshow(same_col, cmap=CMAP, vmin=0, vmax=vmax, aspect="equal")
    im = ax_cross.imshow(cross, cmap=CMAP, vmin=0, vmax=vmax, aspect="equal")

    ax_same.set_yticks(range(n))
    ax_same.set_yticklabels(labels)
    ax_same.set_xticks([0])
    ax_same.set_xticklabels(["same"])
    ax_same.xaxis.tick_top()
    ax_same.tick_params(axis="x", which="major", top=False, labeltop=True)

    ax_cross.set_yticks(range(n))
    ax_cross.set_yticklabels(labels)
    ax_cross.set_xticks(range(n))
    ax_cross.set_xticklabels(labels)
    secax = ax_cross.secondary_xaxis("top")
    secax.set_xticks([(n - 1) / 2.0])
    secax.set_xticklabels(["cross"])
    secax.tick_params(axis="x", which="major", top=False, labeltop=True)

    _annotate_cells(ax_same,  same_col, vmax)
    _annotate_cells(ax_cross, cross,    vmax)

    cax = fig.add_axes([0.84, 0.28, 0.022, 0.50])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Dup out (% of read out)")
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=200)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    plt.close(fig)


def main() -> None:
    denoms = _read_out_denominators([cfg["agent"] for cfg in SYSTEMS.values()])
    for key, cfg in SYSTEMS.items():
        cross = _load_cross_agent(cfg["agent"], cfg["roles"])
        same  = _load_same_agent(cfg["agent"], cfg["roles"])
        _draw_panel(
            title=cfg["title"],
            labels=cfg["labels"],
            cross=cross,
            same=same,
            denom=denoms[cfg["agent"]] or 1.0,
            out_pdf=cfg["out_pdf"],
            out_png=cfg["out_png"],
        )


if __name__ == "__main__":
    main()
