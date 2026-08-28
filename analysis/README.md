# Analysis: paper tables & figures

Everything needed to regenerate the paper's numbers from raw episode logs.

## Data flow

```
logs/<title>/**/*.json                      # one episode JSON per run (from run_matrix.py)
   │
   ├─ scripts/episode_meta.py   ──►  analysis/data/episodes.csv            (1 row / episode)
   ├─ scripts/episode_turns.py  ──►  analysis/data/turns.csv               (1 row / LLM call)
   │     (token-efficiency runs → episodes_tokenefficiency.csv / turns_tokenefficiency.csv)
   ▼
analysis/_common.py             # shared loader + display names + ordering
   ▼
analysis/<generator>.py         ──►  analysis/tables/*.tex , analysis/figures/*.pdf
```

`analysis/data/`, `analysis/tables/`, `analysis/figures/` are generated and
gitignored. Regenerate the CSVs first (see "Regenerating" below), then run the
generators. Never hand-edit numbers — every table/figure comes from a script.

## Generators by paper artifact

Each generator below produces exactly the table(s)/figure(s) the paper
`\input`s; nothing here is dead. (Earlier exploratory generators that fed only
commented-out or never-included artifacts have been removed.)

| Paper artifact                                        | Script                       |
|-------------------------------------------------------|------------------------------|
| Table 1 — main results (Pass / Out tok / Energy)      | `main_table.py`              |
| Table — per-episode energy by difficulty (+ bins)     | `difficulty_bins.py`         |
| Table — retrieval comparison (Lib vs BM25 / SFR)      | `retrieval_comparison.py`    |
| Table — role comparison (compact / BOAD / HyperAgent) | `role_breakdown.py`          |
| Table — energy decomposition (+ diagnostics)          | `energy_decomposition.py`    |
| Table — read-output breakdown                         | `read_output_breakdown.py`   |
| Table — total tokens vs energy                        | `total_vs_energy.py`         |
| Fig — duplicate-output heatmaps (BOAD / HyperAgent)   | `dup_output_heatmap.py`      |
| Fig — Librarian turns per k-th invocation             | `librarian_turns_per_wake.py`|
| Fig — Librarian wakes by difficulty bin               | `librarian_wakes_by_bin_fig.py` (helpers in `librarian_wakes_by_bin.py`) |
| §3 cost analysis — within-episode redundancy          | `redundancy_analysis.py`     |
| Appendix prompt listings (`generated_prompts/`)       | `prepare_prompts_for_latex.py` |
| Paired significance tests (all cost/pass claims)      | `significance/` (own README) |
| Duplicate-output verbatim audit                       | `duplicate_audit/` (own README) |

## Analyses that read raw episodes

Two sub-directories work straight off `logs/` instead of the CSVs, each with
its own README:

| Directory | What it produces |
|---|---|
| `significance/` | Paired significance tests behind the "reduces cost, preserves pass rate" claims — bootstrap CIs, Wilcoxon, McNemar, non-inferiority, plus the leave-one-out ablation tests. |
| `duplicate_audit/` | How much of the duplicate-flagged output is literally re-emitted text, i.e. how loose the paper's upper-bound duplicate count is. |

They resolve matrix log dirs through `EPISODE_LOG_PATH` (colon-separated,
defaults to `<repo>/logs`), so archived main-results runs can live on another
volume:

```bash
export EPISODE_LOG_PATH=/data/librarian_logs:$PWD/logs
```

## Regenerating

```bash
# 1. episode- and turn-level CSVs from a matrix's logs
python scripts/episode_meta.py  logs/qwen36_35b_a3b_500_3agents [...]  --out analysis/data/episodes.csv
python scripts/episode_turns.py logs/qwen36_35b_a3b_500_3agents [...]  --out analysis/data/turns.csv

# 2. tables / figures (run from the repo root)
python analysis/main_table.py
python analysis/difficulty_bins.py
# ... etc.
```

> Note: `episode_meta.py`/`episode_turns.py` emit JSONL by default; convert to the
> CSV column layout `_common.py` expects, or point `_common.py`'s `*_PATH` at your
> generated files. The exact backbone log dirs feeding each table are listed in
> `configs/experiments/README.md`.
