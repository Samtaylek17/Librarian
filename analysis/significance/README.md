# Significance tests

Paired statistical tests behind the paper's "reduces cost, preserves pass
rate" claims. Everything reads raw episode JSON; nothing re-runs an agent.

## Pipeline

```
logs/<matrix>/**/*.json
   │  extract_table45.py            (streams episodes → one record per run)
   ▼
analysis/data/table45_episodes.jsonl
   │
   ├─ stats_table45.py     per-comparison paired tests for the main tables
   ├─ pool_librarian.py    pooled / stratified non-inferiority for the Librarian
   └─ pool_clustered.py    task-clustered pooled test at n=500
```

| Script | What it answers |
|---|---|
| `extract_table45.py` | Builds the tidy per-episode record set (model, family, method, task, passed, out_tok, joules, validity) for every main-results and token-efficiency matrix. |
| `stats_table45.py` | Per comparison against its family base, paired by task over the 500-task set. Energy and output tokens get `mean(base − variant)` with a paired bootstrap 95% CI plus a two-sided Wilcoxon signed-rank; pass rate gets McNemar exact for improvement and a paired non-inferiority Wald test at a 2 pp margin for preservation. Holm correction within each metric family. |
| `pool_librarian.py` | Pools the four `base → +Librarian` cells as independent paired blocks, giving a properly powered non-inferiority estimate plus the achieved one-sided 95% margin per cell. |
| `pool_clustered.py` | Averages each task's paired pass differences over the four (model, family) settings, then runs one-sample tests over the n=500 independent tasks. Avoids treating the same task under four settings as four independent observations. |
| `ablation_wilcoxon_taskclustered.py` | Leave-one-out ablations (L3–L6). Averages each task's HyperAgent and BOAD paired diffs, then a Wilcoxon over n=100 tasks. |
| `ablation_wilcoxon_pooled_hosts.py` | Same ablations, pooling both hosts into n=200 paired diffs. The task-clustered variant is the conservative one; report it. |

## Running

The ablation logs live in this checkout's `logs/`, the 500-task main-results
matrices usually on a separate volume. Point `EPISODE_LOG_PATH` at both.

```bash
export EPISODE_LOG_PATH=/data/librarian_logs:$PWD/logs

python analysis/significance/extract_table45.py      # → analysis/data/table45_episodes.jsonl
python analysis/significance/stats_table45.py
python analysis/significance/pool_librarian.py
python analysis/significance/pool_clustered.py
python analysis/significance/ablation_wilcoxon_taskclustered.py
```

Needs `numpy` + `scipy`. Each script takes the episodes JSONL as an optional
first argument; the default is `analysis/data/table45_episodes.jsonl`.

Only episodes with `eval_error is None` and at least one LLM call count as
valid, and every comparison keeps only tasks valid in **both** configs.
