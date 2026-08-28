# Experiment matrices

Each YAML is a run matrix consumed by `scripts/run_matrix.py`. The main
experiments use **SWE-Bench Verified (500 tasks)** and the two Qwen3.6
backbones (`Qwen/Qwen3.6-35B-A3B-FP8`, `Qwen/Qwen3.6-27B-FP8`).

```
example.yaml          Minimal template (start here).

main_results/         One full-matrix config per backbone (500 tasks each):
                        qwen36_35b_a3b_500.yaml   Qwen3.6-35B-A3B
                        qwen36_27b_500.yaml       Qwen3.6-27B
                      Each runs all BOAD/HyperAgent variants — base, +Librarian,
                      +LastNObservation, +Caveman, +Caveman+Librarian — plus the
                      vanilla single-agent difficulty reference (11 agents × 500).
                      The main table, the difficulty stratification, the
                      role-token / turns-per-wake analyses, AND the paired
                      significance tests are all derived from these runs by the
                      scripts in analysis/.

retrieval/            Retrieval comparison (Librarian vs BM25 vs SFR-dense),
                      one per backbone:
                        qwen36_35b_a3b_retrieval.yaml   Qwen3.6-35B-A3B
                        qwen36_27b_retrieval.yaml       Qwen3.6-27B
                      The retrieval table samples 100 common tasks from these runs
                      + the main-results Librarian runs (analysis/retrieval_comparison.py).
                      The `sfr_code` backend needs the SFR embedding server
                      (scripts/serving/sfr_embed_serve.sh); set each run's
                      `retrieval_endpoint` (see RETRIEVAL_ENDPOINT in .env.example).

librarian_ablation_100.yaml
                      Leave-one-out ablations of the Librarian's four mechanisms —
                      L3 nopersist / L4 nopointer / L5 noscope / L6 nofresh — on
                      hyperagent_librarian + boad_librarian (8 agents). 100 tasks
                      drawn from the 500-task universe with
                      random.Random(42).sample(sorted(ids), 100), seed 0, Qwen3.6-27B.
                      Base settings match main_results so L1/L2 numbers stay
                      comparable. Tests: analysis/significance/ablation_wilcoxon_*.py.

exec_librarian_persist_100.yaml
                      Does the Librarian's mechanism generalize past code navigation?
                      HyperAgent with ONLY the Executor made persistent
                      (hyperagent_exec_librarian): verbatim upstream Executor prompt
                      plus a persistence section, no pointer channel, no scope
                      additions, no freshness report. Same 100 tasks / seed 0 / base
                      as librarian_ablation_100, so the comparison is paired.

live_verified_100.yaml
                      Cross-benchmark generalization: SWE-bench-Live verified split,
                      100 tasks sampled with random.Random(42).sample(sorted(ids), 100).
                      HyperAgent / BOAD baselines vs their Librarian variants, seed 0,
                      base matching librarian_ablation_100.

smoke_baselines.yaml  End-to-end smoke: every non-RAG baseline once on one task.
                      Run after touching agent code, before dispatching a matrix.
librarian_ablation_smoke.yaml
                      Same idea for the 8 ablation agents, on one fast task.
```

`logs/<title>/` holds the raw episode JSON each matrix produces; the analysis
scripts in `analysis/` read those logs. See the top-level `README.md` for the
end-to-end reproduction flow.
