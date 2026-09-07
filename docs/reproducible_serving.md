# Reproducible Qwen3.6 serving

The repository's agent/evaluation environment and its model-serving environment
are intentionally separate. Do not install or upgrade vLLM in `.venv`. The old
vLLM 0.9.2 stack cannot load Qwen3.6 and does not register `qwen3_xml`.

## Frozen inputs

`scripts/serving/reproduction_lock.json` is the source of truth for:

- vLLM 0.19.1 on `linux/amd64`, with the official base image pinned by digest;
- the `qwen3_xml` parser used by the published launcher;
- full model and tokenizer revisions for both Qwen3.6 FP8 backbones;
- the disclosure string `reconstructed reproduction using vLLM 0.19.1`.

If the original authors supply their exact image and checkpoint revisions,
replace the lock and Dockerfile base together, rebuild, rerun validation, and
start formal collection in a new log directory. Never relabel the reconstructed
environment as bit-exact.

## Build and distribute once

```bash
scripts/serving/build_vllm_image.sh
docker image inspect librarian-vllm:0.19.1 --format '{{.Id}}'
```

Build one image, then move that exact artifact to every server through a
content-addressed registry or `docker save`/`docker load`. Do not independently
modify or rebuild the image after collection begins. The launcher resolves the
tag to its immutable local image ID, passes that identity to the server, and
the matrix runner requires all workers to report the same fingerprint.
Commit the experiment code and start from a clean working tree; the repository
commit and dirty state are included in the fingerprint.

The default checkpoint and compile caches live under the repository's ignored
`.cache/` directory, which should be on the large `/raid` filesystem. Override
them only when necessary:

```bash
export VLLM_HF_CACHE=/raid/model-cache/huggingface
export VLLM_CACHE=/raid/model-cache/vllm
```

## Start one process per GPU

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_PORT=8000 \
  scripts/serving/vllm_serve_energy.sh

CUDA_VISIBLE_DEVICES=1 VLLM_PORT=8001 \
  scripts/serving/vllm_serve_energy.sh
```

Set `VLLM_MODEL=Qwen/Qwen3.6-27B-FP8` for the 27B matrices. Each process must
see exactly one GPU. The launcher preserves the experiment's 131,072-token
context, FP8 model, prefix caching, memory utilization, and parser flags. Its
stdout/stderr is also retained under `logs/serving/` for anomaly diagnosis.

If the host driver is too old for the image CUDA runtime, the A100 supports the
official image's compatibility path; set `VLLM_ENABLE_CUDA_COMPATIBILITY=1`
uniformly on every worker and restart validation. That setting becomes part of
the frozen deployment policy even though the actual CUDA/driver versions are
already fingerprinted.

## Validate, then run

Run the non-reportable protocol against every worker:

```bash
scripts/serving/validate_vllm_server.py \
  --base-url http://localhost:8000/v1 \
  --expected-model Qwen/Qwen3.6-35B-A3B-FP8
```

It warms the server while checking ordinary generation, single and parallel
tool parsing, a tool-result round trip, energy headers/counter monotonicity,
one-GPU visibility, and prefix-cache reset. Keep validation output separate
from formal logs. The final validation step resets the prefix cache.

Start the unchanged matrix, retaining its declared YAML order:

```bash
python scripts/run_matrix.py \
  --spec configs/experiments/main_results/qwen36_35b_a3b_500.yaml \
  --urls http://localhost:8000/v1,http://localhost:8001/v1 \
  --reset-prefix-cache
```

Formal runs always require `/experiment-info`; `--no-probe` is limited to dry
runs. Dispatch rejects a dirty or unidentified repository checkout. The
manifest records image, checkpoint, library, CUDA, driver, GPU, power-limit,
clock, launch-argument, and repository provenance for every URL. If a
fingerprint changes, use a new log directory and restart the entire affected
matrix. Never merge the cohorts.

Keep one serial dispatcher worker per dedicated GPU, prevent unrelated work on
those GPUs, and hold driver/power/clock policy constant. Operational failures
must be retried as infrastructure failures, not counted as task outcomes.
