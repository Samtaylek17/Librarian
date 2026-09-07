#!/bin/bash
# Build the pinned serving image once, then distribute this exact image to all
# experiment hosts (registry digest or `docker save`/`docker load`).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
IMAGE_TAG="${VLLM_IMAGE:-librarian-vllm:0.19.1}"

docker build \
    --pull \
    --platform linux/amd64 \
    --file "$REPO_ROOT/scripts/serving/Dockerfile.vllm" \
    --tag "$IMAGE_TAG" \
    "$REPO_ROOT/scripts/serving"

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE_TAG")"
ACTUAL_VERSION="$(
    docker run --rm --entrypoint python3 "$IMAGE_ID" \
        -c 'import importlib.metadata; print(importlib.metadata.version("vllm"))'
)"
if [ "$ACTUAL_VERSION" != "0.19.1" ]; then
    echo "ERROR: built image contains vLLM $ACTUAL_VERSION, expected 0.19.1"
    exit 1
fi
echo "Built $IMAGE_TAG"
echo "Immutable local image ID: $IMAGE_ID"
echo "Use this same image ID/digest on every formal-run host."
