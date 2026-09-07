#!/bin/bash
# Launch the immutable vLLM 0.19.1 experiment container on one GPU.
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
#   VLLM_IMAGE  librarian-vllm:0.19.1       image built by build_vllm_image.sh
#   VLLM_HF_CACHE <repo>/.cache/huggingface checkpoint cache on large storage
#   VLLM_CACHE    <repo>/.cache/vllm        persistent compile cache
#   VLLM_TMPDIR   <repo>/.cache/tmp         bash / docker / vLLM scratch
#                                           (do not use OS /tmp on shared nodes)

set -euo pipefail

if [ -z "${VLLM_PORT:-}" ]; then
    echo "ERROR: VLLM_PORT not set. Use VLLM_PORT=<port> $0"
    exit 1
fi

REPO_ROOT="${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
VLLM_IMAGE="${VLLM_IMAGE:-librarian-vllm:0.19.1}"
VLLM_HF_CACHE="${VLLM_HF_CACHE:-$REPO_ROOT/.cache/huggingface}"
VLLM_CACHE="${VLLM_CACHE:-$REPO_ROOT/.cache/vllm}"
# Shared-node OS disks are often full; here-docs and container /tmp must
# live on the same volume as the repo (typically /raid), not /.
VLLM_TMPDIR="${VLLM_TMPDIR:-$REPO_ROOT/.cache/tmp}"
export TMPDIR="$VLLM_TMPDIR"
LOCK_FILE="$REPO_ROOT/scripts/serving/reproduction_lock.json"
mkdir -p "$VLLM_HF_CACHE" "$VLLM_CACHE" "$VLLM_TMPDIR"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES to exactly one GPU (for example, 0)."
    exit 1
fi

if ! LOCK_OUTPUT="$(python3 - "$LOCK_FILE" "$VLLM_MODEL" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    lock = json.load(f)
revision = lock["models"].get(sys.argv[2])
if revision is None:
    raise SystemExit(f"model {sys.argv[2]!r} is absent from {sys.argv[1]}")
print(revision)
print(lock["vllm_version"])
print(lock["minimum_vllm_version"])
print(lock["tool_call_parser"])
print(lock["reproduction_status"])
PY
)"; then
    exit 1
fi
readarray -t LOCK_VALUES <<<"$LOCK_OUTPUT"
MODEL_REVISION="${LOCK_VALUES[0]}"
LOCKED_VLLM_VERSION="${LOCK_VALUES[1]}"
MINIMUM_VLLM_VERSION="${LOCK_VALUES[2]}"
TOOL_CALL_PARSER="${LOCK_VALUES[3]}"
REPRODUCTION_STATUS="${LOCK_VALUES[4]}"

if ! IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$VLLM_IMAGE" 2>/dev/null)"; then
    echo "ERROR: serving image $VLLM_IMAGE is not built locally."
    echo "Run: scripts/serving/build_vllm_image.sh"
    exit 1
fi

mkdir -p "$REPO_ROOT/logs/serving"
SERVER_LOG="$REPO_ROOT/logs/serving/vllm_${VLLM_PORT}_$(date -u +%Y%m%dT%H%M%SZ).log"
REPO_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]; then
    REPO_DIRTY=1
else
    REPO_DIRTY=0
fi

echo "=== vLLM Server (1 GPU) ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Port: $VLLM_PORT   Model: $VLLM_MODEL"
echo "Revision: $MODEL_REVISION"
echo "Image: $IMAGE_ID"
echo "Server log: $SERVER_LOG"
echo "==========================="

DOCKER_ENV=(
    --env CUDA_VISIBLE_DEVICES=0
    --env VLLM_SERVER_DEV_MODE=1
    --env VLLM_EXPECTED_VERSION="$LOCKED_VLLM_VERSION"
    --env VLLM_MIN_VERSION="$MINIMUM_VLLM_VERSION"
    --env VLLM_EXPECTED_TOOL_PARSER="$TOOL_CALL_PARSER"
    --env VLLM_EXPECTED_MODEL="$VLLM_MODEL"
    --env VLLM_MODEL_REVISION="$MODEL_REVISION"
    --env VLLM_IMAGE_DIGEST="$IMAGE_ID"
    --env LIBRARIAN_GIT_COMMIT="$REPO_COMMIT"
    --env LIBRARIAN_GIT_DIRTY="$REPO_DIRTY"
    --env LIBRARIAN_REPRODUCTION_STATUS="$REPRODUCTION_STATUS"
)
if [ -n "${HF_TOKEN:-}" ]; then
    DOCKER_ENV+=(--env HF_TOKEN)
fi
if [ -n "${VLLM_ENABLE_CUDA_COMPATIBILITY:-}" ]; then
    DOCKER_ENV+=(--env VLLM_ENABLE_CUDA_COMPATIBILITY)
fi

# Resolve the tag to an immutable image ID before launch. Each server is
# single-GPU and foregrounded; tee preserves a persistent diagnostic log.
set +e
docker run --rm \
    --name "librarian-vllm-$VLLM_PORT" \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    --ipc host \
    --publish "$VLLM_PORT:8000" \
    --volume "$VLLM_HF_CACHE:/root/.cache/huggingface" \
    --volume "$VLLM_CACHE:/root/.cache/vllm" \
    --volume "$VLLM_TMPDIR:/tmp" \
    "${DOCKER_ENV[@]}" \
    "$IMAGE_ID" \
    serve "$VLLM_MODEL" \
    --revision "$MODEL_REVISION" \
    --tokenizer-revision "$MODEL_REVISION" \
    --port 8000 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --dtype auto \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" 2>&1 | tee -a "$SERVER_LOG"
status=${PIPESTATUS[0]}
set -e
exit "$status"
