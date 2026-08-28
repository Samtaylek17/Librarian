from __future__ import annotations

import logging
import time
from typing import Any

import httpx
import numpy as np

from eval_pipeline.agents.retrieval.base import RAGCorpus, RAGHit

logger = logging.getLogger(__name__)


class DenseHTTPCorpus(RAGCorpus):

    def __init__(
        self,
        *,
        endpoint: str,
        timeout: float = 15.0,
        max_retries: int = 3,
        passage_separator: str = "\n---\n",
    ) -> None:
        if not endpoint:
            raise ValueError(
                "DenseHTTPCorpus requires retrieval_endpoint (e.g. "
                "http://localhost:19001) — none provided."
            )
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.passage_separator = passage_separator
        self._client = httpx.Client(timeout=timeout)
        self._commands: list[str] = []
        self._observations: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._dim: int | None = None

    def __len__(self) -> int:
        return len(self._commands)

    def close(self) -> None:
        self._commands.clear()
        self._observations.clear()
        self._embeddings.clear()
        try:
            self._client.close()
        except Exception:
            pass

    def _post_embed(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.post(
                    f"{self.endpoint}/embed",
                    json={"texts": texts, "is_query": is_query},
                )
                if 400 <= r.status_code < 500:
                    r.raise_for_status()
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"server {r.status_code}", request=r.request,
                        response=r,
                    )
                body = r.json()
                arr = np.asarray(body["embeddings"], dtype=np.float32)
                if self._dim is None:
                    self._dim = int(arr.shape[1])
                if attempt > 0:
                    logger.info(
                        "embed succeeded on retry %d/%d",
                        attempt, self.max_retries,
                    )
                return arr
            except httpx.HTTPStatusError as e:
                if e.response is not None and 400 <= e.response.status_code < 500:
                    raise
                last_exc = e
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exc = e
            except Exception as e:
                last_exc = e
            if attempt < self.max_retries:
                backoff = 2 ** attempt
                logger.warning(
                    "embed attempt %d/%d failed (%s); retrying in %ds",
                    attempt + 1, self.max_retries + 1, last_exc, backoff,
                )
                time.sleep(backoff)
        assert last_exc is not None
        logger.error(
            "embed gave up after %d attempts: %s",
            self.max_retries + 1, last_exc,
        )
        raise last_exc

    def add(self, command: str, observation: str) -> None:
        if not observation:
            return
        doc = command + self.passage_separator + observation
        try:
            emb = self._post_embed([doc], is_query=False)[0]
        except Exception as e:
            logger.warning(
                "DenseHTTPCorpus.add: embed failed (%s) — skipping pair", e,
            )
            return
        self._commands.append(command)
        self._observations.append(observation)
        self._embeddings.append(emb)

    def retrieve(
        self,
        query: str,
        *,
        threshold: float,
        max_pairs: int,
    ) -> list[RAGHit]:
        if max_pairs <= 0 or not self._embeddings:
            return []
        try:
            q_emb = self._post_embed([query], is_query=True)[0]
        except Exception as e:
            logger.warning(
                "DenseHTTPCorpus.retrieve: query embed failed (%s) — "
                "returning empty hits", e,
            )
            return []
        mat = np.stack(self._embeddings)
        scores = mat @ q_emb
        order = np.argsort(scores)[::-1]
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
