"""Trajectory log — a flat list of :class:`Turn` records.

A *turn* (see ``runtime/turn.py``) bundles one agent loop step:
identity + LLM call + observation(s). This module accumulates turns and
serialises them.

The class is still named ``LLMCallLog`` (and exposes ``to_dicts()`` /
``total()``) so the six runner construction sites and ``AgentRunner.run``
need no change. New code calls :meth:`LLMCallLog.record_turn`; the legacy
:meth:`record` kwargs API is kept as a thin shim that maps old
``LLMCallRecord`` fields onto a ``Turn`` (used by BOAD's one-shot plan
call until it is agent-ified in Phase 4).

``LLMCallRecord`` is retained for backward-compatible imports only; it is
no longer the storage unit.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from eval_pipeline.agents.runtime.turn import (
    ExecStats,
    LLMStats,
    Observation,
    Turn,
)

# Field names that belong to LLMStats (everything else passed to the legacy
# record() shim is treated as turn-level identity/content).
_LLM_STATS_FIELDS = frozenset(LLMStats.__dataclass_fields__)


@dataclass
class LLMCallRecord:
    """Legacy per-call record. Kept only for backward-compatible imports
    (``eval_pipeline.agents.observability.LLMCallRecord`` /
    ``openhands.LLMCallRecord``). No longer used for storage — see
    :class:`Turn`."""

    agent_id: str = ""
    agent_role: str = ""
    input: list[dict] = field(default_factory=list)
    text: str = ""
    action: list[dict] = field(default_factory=list)
    observation: list[dict] = field(default_factory=list)
    raw_output: list[dict] = field(default_factory=list)

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    duration_seconds: float = 0.0
    n_attempts: int = 1
    joules: float | None = None
    joules_per_gpu: list[float] = field(default_factory=list)
    joules_raw: float | None = None
    joules_per_gpu_raw: list[float] = field(default_factory=list)
    idle_power_W: list[float] = field(default_factory=list)
    cache_q_delta: float | None = None
    cache_h_delta: float | None = None
    cache_hit_ratio: float | None = None
    started_at: float | None = None
    ended_at: float | None = None
    error: str | None = None


class LLMCallLog:

    def __init__(self, live_dump_path: str | None = None) -> None:
        self.turns: list[Turn] = []
        self.live_dump_path = live_dump_path
        self._live_last_flush = 0.0
        self._live_last_bytes = 0
        if self.live_dump_path:
            try:
                os.makedirs(os.path.dirname(self.live_dump_path), exist_ok=True)
                open(self.live_dump_path, "w").close()
            except Exception:
                pass

    # ── primary API ─────────────────────────────────────────────────
    def record_turn(self, turn: Turn) -> Turn:
        self.turns.append(turn)
        self._rewrite_live_dump()
        return turn

    def update_live(self) -> None:
        """Re-flush the live dump after an in-place edit (e.g. an
        observation appended to the open turn)."""
        self._rewrite_live_dump()

    # ── legacy shim (boad one-shot plan call) ───────────────────────
    def record(self, **kwargs: Any) -> Turn:
        stats_kwargs = {
            k: v for k, v in kwargs.items() if k in _LLM_STATS_FIELDS
        }
        obs = [
            Observation(
                tool_name=o.get("tool_name", ""),
                payload=str(o.get("output", "")),
                exec_stats=ExecStats(output_len=len(str(o.get("output", "")))),
            )
            for o in (kwargs.get("observation") or [])
            if isinstance(o, dict)
        ]
        turn = Turn(
            agent_id=kwargs.get("agent_id", ""),
            agent_type=kwargs.get("agent_role", ""),
            llm_input=kwargs.get("input", []) or [],
            llm_output_text=kwargs.get("text", "") or "",
            raw_output=kwargs.get("raw_output", []) or [],
            parsed_actions=kwargs.get("action", []) or [],
            llm_stats=LLMStats(**stats_kwargs),
            observations=obs,
        )
        return self.record_turn(turn)

    # ── serialisation ───────────────────────────────────────────────
    def _rewrite_live_dump(self) -> None:
        if not self.live_dump_path:
            return
        # A rewrite serialises EVERY turn (an old turn can still mutate:
        # a MAS orchestrator's delegation turn binds its observation after
        # the whole sub-agent session), so per-call rewriting is O(n^2) in
        # episode length. Space rewrites proportionally to the file size to
        # cap sustained live-dump IO at ~5 MB/s per episode.
        now = time.time()
        min_interval = max(10.0, self._live_last_bytes / 5e6)
        if now - self._live_last_flush < min_interval:
            return
        try:
            lines = []
            for t in self.turns:
                d = asdict(t)
                li = d.get("llm_input")
                if isinstance(li, list) and li:
                    d["llm_input"] = (
                        f"<omitted in live dump: {len(li)} messages; "
                        "full copy lands in the final episode JSON>"
                    )
                lines.append(json.dumps(d, default=str) + "\n")
            with open(self.live_dump_path, "w") as f:
                f.writelines(lines)
                f.flush()
            self._live_last_bytes = sum(len(ln) for ln in lines)
            self._live_last_flush = now
        except Exception:
            pass

    def to_dicts(self) -> list[dict[str, Any]]:
        return [asdict(t) for t in self.turns]

    def total(self) -> dict[str, int]:
        return {
            "input_tokens": sum(t.llm_stats.input_tokens for t in self.turns),
            "output_tokens": sum(t.llm_stats.output_tokens for t in self.turns),
            "cached_tokens": sum(t.llm_stats.cached_tokens for t in self.turns),
        }
