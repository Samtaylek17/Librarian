"""BOAD-Librarian-RAG: BOAD MAS + fresh-spawn librarian + query-driven RAG.

Same manipulation as :mod:`mas.librarian_rag` applied to
:class:`BOADLibrarianMASRunner`: swap the persistent
:class:`SharedSubAgent` for a :class:`RAGNavigator`. Everything else
(fixed 6-step plan, issue_analyzer sub-agent, librarian-routed bash,
``submit_result(result, view_commands)`` side-channel, view-only
sub-agent tools, ``_LIBRARIAN_SYSTEM`` prompt) is inherited verbatim.

Backend-agnostic via ``retrieval_backend``; ``sfr_code`` needs an
``retrieval_endpoint``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from eval_pipeline.agents.bases.openhands_base import build_openhands_agent
from eval_pipeline.agents.mas.boad_librarian import (
    BOADLibrarianConfig,
    BOADLibrarianMASRunner,
)
from eval_pipeline.agents.retrieval import RAGNavigator, build_corpus

logger = logging.getLogger(__name__)


@dataclass
class BOADLibrarianRAGConfig(BOADLibrarianConfig):

    retrieval_backend: str = "bm25"
    retrieval_threshold: float = 0.0
    retrieval_max_pairs: int = 5
    retrieval_endpoint: str | None = None


class BOADLibrarianRAGMASRunner(BOADLibrarianMASRunner):

    def __init__(
        self,
        *,
        retrieval_backend: str = BOADLibrarianRAGConfig.retrieval_backend,
        retrieval_threshold: float = BOADLibrarianRAGConfig.retrieval_threshold,
        retrieval_max_pairs: int = BOADLibrarianRAGConfig.retrieval_max_pairs,
        retrieval_endpoint: str | None = BOADLibrarianRAGConfig.retrieval_endpoint,
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
                "[BOADLibrarianRAG] corpus backend=%s endpoint=%s "
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
