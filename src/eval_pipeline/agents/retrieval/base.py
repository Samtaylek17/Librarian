from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RAGHit:
    """One retrieval hit.

    ``command``: the str_replace_editor verb + args that produced the
    observation (e.g. ``view /path:120-160``). Carried so the prompt
    can show provenance without re-running the tool.
    ``observation``: the verbatim tool output the model saw at the time.
    ``score``: backend-specific relevance score (BM25 sum / cosine /
    rerank). Higher = more relevant. Comparable WITHIN one corpus only.
    """

    command: str
    observation: str
    score: float


class RAGCorpus(ABC):

    @abstractmethod
    def add(self, command: str, observation: str) -> None:
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        threshold: float,
        max_pairs: int,
    ) -> list[RAGHit]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    def close(self) -> None:
        return None
