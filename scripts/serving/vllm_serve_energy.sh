#!/bin/bash
# Launch vLLM with the energy + reset-prefix-cache wrapper, on one GPU.
#
# This is the only serving entry point the evaluation pipeline needs:
#   * /energy             — per-GPU cumulative mJ from NVML (energy measurement)
#   * /reset_prefix_cache — requires VLLM_SERVER_DEV_MODE=1 (set below)
#   * /v1/...             — OpenAI-compatible chat completions
#
# Run one instance per GPU; give each its own VLLM_PORT and pass the resulting
# URLs to scripts/run_matrix.py --urls.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 VLLM_PORT=8000 scripts/serving/vllm_serve_energy.sh
#
# Env (all optional unless noted, shown with defaults):
#   VLLM_PORT   (required)                  OpenAI API port to serve on
#   VLLM_MODEL  Qwen/Qwen3.6-35B-A3B-FP8    model to serve (paper backbones:
#                                           Qwen3.6-35B-A3B-FP8 / Qwen3.6-27B-FP8)
#   REPO_ROOT   <git toplevel>              path to this repository
#   CONDA_BASE  unset                       conda install holding the vllm env;
#                                           set it to have this script activate
#   VLLM_ENV    vllm                        conda env name with vllm installed

set -euo pipefail

if [ -z "${VLLM_PORT:-}" ]; then
    echo "ERROR: VLLM_PORT not set. Use VLLM_PORT=<port> $0"
    exit 1
fi

REPO_ROOT="${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"

echo "=== vLLM Server (1 GPU) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Port: $VLLM_PORT   Model: $VLLM_MODEL"
echo "==========================="

# Activate an env that has vllm + this package installed. Skipped when
# CONDA_BASE is unset — run the script from an already-activated env instead.
if [ -n "${CONDA_BASE:-}" ]; then
    source "${CONDA_BASE}/bin/activate" "${VLLM_ENV:-vllm}"
fi

# VLLM_SERVER_DEV_MODE=1 exposes POST /reset_prefix_cache (404 otherwise),
# which the matrix runner calls between episodes for joules comparability.
export VLLM_SERVER_DEV_MODE=1

# Capture API port + unset VLLM_PORT so vLLM internal port allocation uses
# OS-assigned ports (avoids dist-init port collision when multiple vLLM
# instances co-locate on one node).
API_PORT="$VLLM_PORT"
unset VLLM_PORT

# Wrapper monkey-patches vLLM's FastAPI app to expose /energy on the same
# port. Argv after the script path is forwarded verbatim to `vllm serve`.
exec python "$REPO_ROOT/scripts/serving/vllm_with_energy.py" \
    serve "$VLLM_MODEL" \
    --port "$API_PORT" \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --dtype auto \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
