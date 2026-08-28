from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from eval_pipeline.agents.bases.openhands_base import build_openhands_agent
from eval_pipeline.agents.mas.librarian import (
    LibrarianConfig,
    LibrarianMASRunner,
)
from eval_pipeline.agents.retrieval import RAGNavigator, build_corpus

logger = logging.getLogger(__name__)


@dataclass
class LibrarianRAGConfig(LibrarianConfig):

    retrieval_backend: str = "bm25"
    retrieval_threshold: float = 0.0
    retrieval_max_pairs: int = 5
    retrieval_endpoint: str | None = None


class LibrarianRAGMASRunner(LibrarianMASRunner):
    """LibrarianMASRunner with RAGNavigator instead of SharedSubAgent.

    All wiring outside the navigator wrap is inherited verbatim
    (main-agent prompt, librarian-routed bash, str_replace_editor,
    submit_result side-channel, OH system prompt). Override hooks:

      - ``__init__``: extra kwargs for retrieval backend / threshold /
        max_pairs. Episode-scoped corpus is created lazily at the start
        of every ``run_with_executor`` via ``self._corpus``.
      - ``_make_navigator``: swap SharedSubAgent for RAGNavigator.
    """

    def __init__(
        self,
        *,
        base_agent_builder=build_openhands_agent,
        base_name: str = "oh",
        retrieval_backend: str = LibrarianRAGConfig.retrieval_backend,
        retrieval_threshold: float = LibrarianRAGConfig.retrieval_threshold,
        retrieval_max_pairs: int = LibrarianRAGConfig.retrieval_max_pairs,
        retrieval_endpoint: str | None = LibrarianRAGConfig.retrieval_endpoint,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_agent_builder=base_agent_builder,
            base_name=base_name,
            **kwargs,
        )
        self.retrieval_backend = retrieval_backend
        self.retrieval_threshold = retrieval_threshold
        self.retrieval_max_pairs = retrieval_max_pairs
        self.retrieval_endpoint = retrieval_endpoint
        self._corpus = None
        self._corpus_first_wake = True

    def _make_navigator(self, *, navigator_agent, work_dir: str):
        if self._corpus is None:
            self._corpus = build_corpus(
                self.retrieval_backend,
                endpoint=self.retrieval_endpoint,
            )
            logger.info(
                "[LibrarianRAG] corpus backend=%s endpoint=%s "
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
