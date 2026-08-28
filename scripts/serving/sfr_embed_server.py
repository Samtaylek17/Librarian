#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("sfr_embed")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def _snapshot_and_patch(model_id: str) -> str:
    local_dir = os.path.join(
        os.environ.get("SFR_LOCAL_DIR", "/tmp/sfr_embed_snapshot"),
        model_id.replace("/", "__"),
    )
    logger.info("snapshot_download %s -> %s", model_id, local_dir)
    snapshot_download(model_id, local_dir=local_dir)
    mg = os.path.join(local_dir, "modeling_gemma2.py")
    if not os.path.isfile(mg):
        logger.warning("modeling_gemma2.py not found at %s — skipping patch", mg)
        return local_dir
    src = open(mg).read()
    needle = (
        'Gemma2Model.from_pretrained(config._name_or_path, '
        'trust_remote_code=True, is_causal=False, device_map="auto")'
    )
    replacement = (
        'Gemma2Model.from_pretrained(config._name_or_path, '
        'trust_remote_code=True, is_causal=False)'
    )
    if needle in src:
        open(mg, "w").write(src.replace(needle, replacement))
        logger.info("patched modeling_gemma2.py — removed device_map='auto'")
    elif replacement in src:
        logger.info("modeling_gemma2.py already patched (no device_map)")
    else:
        logger.warning(
            "modeling_gemma2.py does not match expected needle — "
            "patch SKIPPED (may need rewriting if model card changes)"
        )
    return local_dir


QUERY_INSTRUCTION = (
    "Instruct: Given Code or Text, retrieval relevant content\nQuery: "
)


class EmbedRequest(BaseModel):
    texts: List[str]
    is_query: bool = False


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    n: int
    dim: int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--model", default="Salesforce/SFR-Embedding-Code-2B_R",
    )
    parser.add_argument(
        "--max-batch", type=int, default=16,
        help="Batch size used by model.encode(); reduce on OOM.",
    )
    parser.add_argument(
        "--max-length", type=int, default=8192,
        help="Tokenizer truncation cap. SFR supports up to 32768.",
    )
    args = parser.parse_args()

    logger.info("Loading %s (bf16) ...", args.model)
    local_dir = _snapshot_and_patch(args.model)
    model = SentenceTransformer(
        local_dir,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = args.max_length
    dim = model.get_sentence_embedding_dimension()
    logger.info(
        "Loaded model: dim=%d, device=%s, cuda=%s",
        dim, model.device, torch.cuda.is_available(),
    )
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(
            "GPU: %s, total_mem=%.1f GB",
            props.name, props.total_memory / 1e9,
        )

    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": args.model,
            "dim": dim,
            "device": str(model.device),
            "cuda": torch.cuda.is_available(),
            "max_length": args.max_length,
            "max_batch": args.max_batch,
        }

    @app.post("/embed", response_model=EmbedResponse)
    def embed(req: EmbedRequest):
        if not req.texts:
            raise HTTPException(status_code=400, detail="texts is empty")
        prompt = QUERY_INSTRUCTION if req.is_query else None
        t0 = time.time()
        embeddings = model.encode(
            req.texts,
            prompt=prompt,
            batch_size=args.max_batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        dt = time.time() - t0
        logger.info(
            "embed n=%d is_query=%s dt=%.3fs (%.1f/s)",
            len(req.texts), req.is_query, dt,
            (len(req.texts) / dt) if dt > 0 else 0.0,
        )
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            n=len(req.texts),
            dim=int(embeddings.shape[1]),
        )

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
