"""HyperAgent-Librarian-RAG: HyperAgent MAS + fresh-spawn librarian + RAG.

Same manipulation as :mod:`mas.librarian_rag` applied to
:class:`HyperAgentLibrarianMASRunner`: swap the persistent
:class:`SharedSubAgent` for a :class:`RAGNavigator`. Planner / Editor /
Executor sub-agents and their delegation tooling are inherited verbatim.

Backend-agnostic via ``retrieval_backend``; ``sfr_code`` needs an
``retrieval_endpoint``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from eval_pipeline.agents.bases.openhands_base import build_openhands_agent
from eval_pipeline.agents.mas.hyperagent_librarian import (
    HyperAgentLibrarianConfig,
    HyperAgentLibrarianMASRunner,
)
from eval_pipeline.agents.retrieval import RAGNavigator, build_corpus

logger = logging.getLogger(__name__)


@dataclass
class HyperAgentLibrarianRAGConfig(HyperAgentLibrarianConfig):

    retrieval_backend: str = "bm25"
    retrieval_threshold: float = 0.0
    retrieval_max_pairs: int = 5
    retrieval_endpoint: str | None = None


class HyperAgentLibrarianRAGMASRunner(HyperAgentLibrarianMASRunner):

    def __init__(
        self,
        *,
        retrieval_backend: str = HyperAgentLibrarianRAGConfig.retrieval_backend,
        retrieval_threshold: float = HyperAgentLibrarianRAGConfig.retrieval_threshold,
        retrieval_max_pairs: int = HyperAgentLibrarianRAGConfig.retrieval_max_pairs,
        retrieval_endpoint: str | None = HyperAgentLibrarianRAGConfig.retrieval_endpoint,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.retrieval_backend = retrieval_backend
        self.retrieval_threshold = retrieval_threshold
        self.retrieval_max_pairs = retrieval_max_pairs
        self.retrieval_endpoint = retrieval_endpoint
        self._corpus = None

    def _make_navigator(self, *, navigator_agent, work_dir: str):
        if self._corpus is None:
            self._corpus = build_corpus(
                self.retrieval_backend,
                endpoint=self.retrieval_endpoint,
            )
            logger.info(
                "[HyperAgentLibrarianRAG] corpus backend=%s endpoint=%s "
                "threshold=%.3f max_pairs=%d",
                self.retrieval_backend, self.retrieval_endpoint,
                self.retrieval_threshold, self.retrieval_max_pairs,
            )
        return RAGNavigator(
            agent=navigator_agent,
            corpus=self._corpus,
            retrieval_threshold=self.retrieval_threshold,
            retrieval_max_pairs=self.retrieval_max_pairs,
            model=self.model,
        )
