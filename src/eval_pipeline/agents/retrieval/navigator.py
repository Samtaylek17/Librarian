from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agents import Agent

from eval_pipeline.agents._reasoning import strip_reasoning
from eval_pipeline.agents.history.view_state import parse_view_action
from eval_pipeline.agents.retrieval.base import RAGCorpus, RAGHit
from eval_pipeline.agents.runtime import safe_runner_run

logger = logging.getLogger(__name__)


def _format_retrieved_block(hits: list[RAGHit]) -> str:
    if not hits:
        return (
            "[RETRIEVED CONTEXT — none. The corpus has no prior "
            "observations that match this query yet.]"
        )
    lines: list[str] = [
        f"[RETRIEVED CONTEXT — {len(hits)} prior observation(s) the "
        "sub-agent collected on earlier wakes of this episode. Each "
        "block is a verbatim quote of what was seen at the time. "
        "Use them as ground truth if they cover the query; otherwise "
        "run your own tools to fill the gap.]",
    ]
    for i, h in enumerate(hits, 1):
        lines.append("")
        lines.append(f"[{i}] command: {h.command}   (relevance={h.score:.3f})")
        lines.append("observation:")
        lines.append(h.observation)
    return "\n".join(lines)


def _extract_view_pairs(
    delta: list[Any],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    pending: tuple[int, str] | None = None
    for msg in delta:
        if not isinstance(msg, dict):
            continue
        mtype = msg.get("type")
        if mtype == "function_call":
            name = msg.get("name", "")
            if name != "str_replace_editor":
                pending = None
                continue
            try:
                args = json.loads(msg.get("arguments", "{}"))
            except Exception:
                pending = None
                continue
            view = parse_view_action(name, args)
            if view is None:
                pending = None
                continue
            path, start, end = view
            end_s = "EOF" if end is None else str(end)
            label = f"str_replace_editor view {path}:{start}-{end_s}"
            pending = (msg.get("call_id", ""), label)
        elif mtype == "function_call_output" and pending is not None:
            call_id = msg.get("call_id", "")
            if call_id != pending[0]:
                pending = None
                continue
            obs = msg.get("output", "") or ""
            if isinstance(obs, dict):
                obs = obs.get("text", "") or ""
            obs = obs.strip()
            if obs:
                pairs.append((pending[1], obs))
            pending = None
        else:
            if mtype == "message" and pending is not None:
                pending = None
    return pairs


@dataclass
class RAGNavigator:

    agent: Agent
    corpus: RAGCorpus
    retrieval_threshold: float = 0.0
    retrieval_max_pairs: int = 5
    model: str | None = None

    last_wake_n_hits: int = 0
    last_wake_n_added: int = 0
    last_wake_view_files: set = field(default_factory=set)

    def close(self) -> None:
        try:
            self.corpus.close()
        except Exception as e:
            logger.warning("RAGNavigator.close: corpus.close failed: %s", e)
        self.last_wake_view_files.clear()

    async def invoke(
        self,
        request: str,
        caller_role: str,
        *,
        hooks,
        max_turns: int | None = None,
        text_only_policy: Any | None = None,
    ) -> str:
        hits = self.corpus.retrieve(
            request,
            threshold=self.retrieval_threshold,
            max_pairs=self.retrieval_max_pairs,
        )
        self.last_wake_n_hits = len(hits)
        block = _format_retrieved_block(hits)
        user_msg = (
            f"[REQUEST from {caller_role}]:\n{request}\n\n{block}"
        )
        input_items: list = [{"role": "user", "content": user_msg}]

        runner_kwargs: dict[str, Any] = {"hooks": hooks}
        if max_turns is not None:
            runner_kwargs["max_turns"] = max_turns
        if text_only_policy is not None:
            runner_kwargs["text_only_policy"] = text_only_policy

        result = await safe_runner_run(
            self.agent, input_items, **runner_kwargs,
        )
        final_output = result.final_output or ""
        if self.model:
            final_output = strip_reasoning(final_output, self.model)

        delta = result.to_input_list()[len(input_items):]
        pairs = _extract_view_pairs(delta)
        for cmd, obs in pairs:
            self.corpus.add(cmd, obs)
        self.last_wake_n_added = len(pairs)
        logger.info(
            "rag_navigator wake: hits=%d added=%d corpus_size=%d",
            self.last_wake_n_hits, self.last_wake_n_added, len(self.corpus),
        )
        return final_output
