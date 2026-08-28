from __future__ import annotations

from eval_pipeline.agents.retrieval.base import RAGCorpus
from eval_pipeline.agents.retrieval.bm25 import BM25Corpus
from eval_pipeline.agents.retrieval.dense import DenseHTTPCorpus


REGISTERED_NAMES = ("bm25", "sfr_code")


def build_corpus(name: str, **kwargs) -> RAGCorpus:
    n = (name or "").lower().strip()
    if n == "bm25" or not n:
        return BM25Corpus()
    if n == "sfr_code":
        endpoint = kwargs.get("endpoint") or kwargs.get("retrieval_endpoint")
        if not endpoint:
            raise ValueError(
                "Backend 'sfr_code' requires retrieval_endpoint (e.g. "
                "http://localhost:19001) — none provided."
            )
        timeout = float(kwargs.get("timeout") or 60.0)
        return DenseHTTPCorpus(endpoint=endpoint, timeout=timeout)
    raise ValueError(
        f"Unknown retrieval backend {name!r}; "
        f"valid: {', '.join(REGISTERED_NAMES)}"
    )
