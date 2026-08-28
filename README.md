# Long Live the Librarian — code & experiments

Evaluation framework for the paper *"Long Live the Librarian! A Persistent
Search Sub-Agent for Energy-Efficient Multi-Agent Software Engineering
Systems."*

LLM coding agents run in Docker, produce patches, and are scored by the
original benchmark harness. A data pipeline records per-LLM-call tokens
and **Joules** so every method can be compared on energy, not just pass rate.
The **Librarian** is a persistent code-lookup sub-agent that replaces a MAS's
navigator role; we evaluate it on top of two multi-agent systems (BOAD,
HyperAgent) and against two token-efficiency methods (Caveman, LastNObservation)
and two retrieval variants (BM25, SFR-dense). Leave-one-out ablations isolate
its four mechanisms, a SWE-bench-Live matrix checks that the effect survives a
different task distribution, and the same persistence mechanism is applied to
HyperAgent's Executor to see whether it generalizes past code navigation.

## Setup

```bash
conda create -n swe python=3.12 -y && conda activate swe
pip install -e ".[all]"            # SWE-bench Verified + Live (or ".[swe-bench]" / ".[dev]")
cp .env.example .env               # then edit; `set -a; source .env; set +a`
```

Needs Python ≥ 3.10, a Docker daemon, and an OpenAI-compatible
chat-completions endpoint. Agent names are in
`src/eval_pipeline/agents/__init__.py`, benchmarks in
`src/eval_pipeline/benchmarks/__init__.py`.

## Experiments

**Serve vLLM** with the energy endpoint. One GPU per worker.

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_PORT=8000 scripts/serving/vllm_serve_energy.sh
```

The script wraps `scripts/serving/vllm_with_energy.py`, which adds
`GET /energy` (per-GPU cumulative mJ via NVML) on the serving port. Without it
`episode.energy` stays `None`. It also exports `VLLM_SERVER_DEV_MODE=1`, which
`POST /reset_prefix_cache` requires — otherwise joules leak across episodes.

**One episode:**

```bash
python scripts/run.py run \
    --agent boad_librarian --benchmark swe_bench \
    --task-id django__django-14631 \
    --base-url http://localhost:8000/v1 --seed 0
```

**A matrix**, round-robin over workers, resumable (any run whose log dir already
has a result JSON is skipped):

```bash
python scripts/run_matrix.py \
    --spec configs/experiments/main_results/qwen36_35b_a3b_500.yaml \
    --urls http://localhost:8000/v1,http://localhost:8001/v1 \
    --reset-prefix-cache
```

One worker thread per URL, sequential within a thread — never stack two agents
on one vLLM worker, or the batcher thrashes the KV cache and contaminates
joules. Add `--server-pool configs/server_pool.yaml` to route containers to
remote docker daemons.

| Matrix in `configs/experiments/` | What it establishes |
|---|---|
| `main_results/qwen36_{35b_a3b,27b}_500.yaml` | Main table, difficulty stratification, role-token and turns-per-wake analyses |
| `retrieval/qwen36_{35b_a3b,27b}_retrieval.yaml` | Librarian vs BM25 vs SFR-dense |
| `librarian_ablation_100.yaml` | Which of the four mechanisms drives the gain |
| `exec_librarian_persist_100.yaml` | Whether persistence helps a non-navigation role |
| `live_verified_100.yaml` | Whether the effect holds on SWE-bench-Live |

Each writes `logs/<title>/<task>/<agent>_seed<N>/<Evaluator>_….json` with patch,
score, token usage, energy, and per-LLM-call records. Details in
[configs/experiments/README.md](configs/experiments/README.md).

## Analysis

Table and figure generators read episode/turn records extracted from the logs;
the other analyses read `logs/` directly and resolve matrix dirs through
`EPISODE_LOG_PATH`.

```bash
export EPISODE_LOG_PATH=/data/librarian_logs:$PWD/logs

# tables / figures
python scripts/episode_meta.py  logs/<matrix> --out analysis/data/episodes.jsonl
python scripts/episode_turns.py logs/<matrix> --out analysis/data/turns.jsonl
python analysis/main_table.py                    # one generator per paper artifact

# paired significance tests
python analysis/significance/extract_table45.py  # → analysis/data/table45_episodes.jsonl
python analysis/significance/stats_table45.py

# supporting analyses
python analysis/duplicate_audit/dup_upper_bound_audit.py
```

| Directory | Contents |
|---|---|
| `analysis/` | One generator per paper table/figure; `_common.py` holds the loader and the column layout the extractors feed ([README](analysis/README.md)) |
| `analysis/significance/` | Bootstrap CIs, Wilcoxon, McNemar, non-inferiority, ablation tests ([README](analysis/significance/README.md)) |
| `analysis/duplicate_audit/` | How much duplicate-flagged output is literally re-emitted text ([README](analysis/duplicate_audit/README.md)) |

`third_party/` is byte-identical vendored upstream;
`scripts/check_third_party_drift.py` fails on drift.

## License

MIT — see [LICENSE](LICENSE). Files under `third_party/` are verbatim copies
of upstream projects and stay under their own licenses; see [NOTICE](NOTICE).
