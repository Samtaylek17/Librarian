#!/bin/bash
# Serve SFR-Embedding-Code-2B_R as an HTTP /embed endpoint, on one GPU.
# Only needed to reproduce the dense-retrieval (sfr_code) RAG baseline; the
# persistent-Librarian and BM25 runs do not use it.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 EMBED_PORT=9002 scripts/serving/sfr_embed_serve.sh
#
# Env (all optional unless noted, shown with defaults):
#   EMBED_PORT  (required)        port to serve /embed on
#   REPO_ROOT   <git toplevel>    path to this repository
#   CONDA_BASE  unset             conda install holding the vllm env; set it
#                                 to have this script activate
#   VLLM_ENV    vllm              conda env name (also used for embedding)

set -euo pipefail

if [ -z "${EMBED_PORT:-}" ]; then
    echo "ERROR: EMBED_PORT not set. Use EMBED_PORT=<port> $0"
    exit 1
fi

REPO_ROOT="${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || pwd)}"

echo "=== SFR-Embedding-Code-2B_R Server (1 GPU) ==="
echo "Node: $(hostname)   Port: $EMBED_PORT"
echo "=============================================="

if [ -n "${CONDA_BASE:-}" ]; then
    source "${CONDA_BASE}/bin/activate" "${VLLM_ENV:-vllm}"
fi

# sentence-transformers + accelerate may not be in the serving env; install
# to per-user site-packages on demand (idempotent). accelerate is required
# because SFR's modeling code uses device_map="auto" in from_pretrained.
python -c "import sentence_transformers, accelerate" 2>/dev/null \
    || pip install --user --quiet sentence-transformers accelerate

exec python "$REPO_ROOT/scripts/serving/sfr_embed_server.py" \
    --port "$EMBED_PORT" \
    --model Salesforce/SFR-Embedding-Code-2B_R \
    --max-batch 16 \
    --max-length 8192
