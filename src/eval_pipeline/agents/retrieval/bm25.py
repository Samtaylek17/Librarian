from __future__ import annotations

import re
from typing import Final

from rank_bm25 import BM25Okapi

from eval_pipeline.agents.retrieval.base import RAGCorpus, RAGHit


_TOKEN_RX: Final = re.compile(r"[A-Za-z0-9_.]+|\S")
_FRAG_SPLIT_RX: Final = re.compile(r"[.\:/]+")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    for tok in _TOKEN_RX.findall(text):
        out.append(tok)
        if "." in tok or ":" in tok or "/" in tok:
            for frag in _FRAG_SPLIT_RX.split(tok):
                if frag:
                    out.append(frag)
    return out


class BM25Corpus(RAGCorpus):

    def __init__(self) -> None:
        self._commands: list[str] = []
        self._observations: list[str] = []
        self._docs: list[list[str]] = []

    def add(self, command: str, observation: str) -> None:
        if not observation:
            return
        self._commands.append(command)
        self._observations.append(observation)
        doc = _tokenize(command) + _tokenize(observation)
        self._docs.append(doc)

    def __len__(self) -> int:
        return len(self._commands)

    def close(self) -> None:
        self._commands.clear()
        self._observations.clear()
        self._docs.clear()

    def retrieve(
        self,
        query: str,
        *,
        threshold: float,
        max_pairs: int,
    ) -> list[RAGHit]:
        if max_pairs <= 0 or not self._docs:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        index = BM25Okapi(self._docs)
        scores = index.get_scores(q_tokens)
        order = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True,
        )
        hits: list[RAGHit] = []
        for i in order:
            if len(hits) >= max_pairs:
                break
            s = float(scores[i])
            if s < threshold:
                break
            hits.append(
                RAGHit(
                    command=self._commands[i],
                    observation=self._observations[i],
                    score=s,
                )
            )
        return hits
