"""First-class *turn* records.

A **turn** is one cycle of an agent's loop:

  agent_id + agent_type
    → LLM call (the exact input sent, the output, the parsed action(s),
      and per-call statistics)
    → Observation(s) (the result of executing each parsed action, plus
      per-execution statistics)

This replaces the old flat ``LLMCallRecord`` list (see
``observability/call_log.py``), whose ``observation`` field was bound
*lazily* on the next ``on_llm_start`` and which discarded per-tool
execution statistics. A ``Turn`` bundles everything for one loop step and
is closed *synchronously* when its action's observation is produced.

Turns are stored as a **flat list** (``EpisodeData.turns``); the call
tree is reconstructed from ``Turn.parent_agent_id`` and
``Observation.child_agent_id`` links rather than physical nesting.

Pure data — no behaviour. All measurement fields carry the same meaning
as the legacy record so the energy / token extractors keep
working after they switch their input source to ``turns[].llm_stats``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMStats:
    """Per-LLM-call measurements. Carried verbatim from the legacy
    ``LLMCallRecord`` so downstream extractors are a field-rename, not a
    semantic change."""

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


@dataclass
class ExecStats:
    """Per-action execution measurements.

    Sourced from the tool wrappers (bash / str_replace_editor / submit /
    agent-call), which already carry ``exit_code`` etc. in the
    ``_LAST_TOOL_EVENT_PAIR`` contextvar — previously discarded.
    """

    exit_code: int | None = None
    duration_seconds: float = 0.0
    output_len: int = 0
    truncated: bool = False


@dataclass
class Observation:
    """The result of executing one parsed action.

    ``child_agent_id`` is set only when the action was an agent-call tool;
    it links this observation to the spawned/invoked agent's own turns,
    making the call tree reconstructable from a flat turn list.
    """

    tool_name: str = ""
    args: dict = field(default_factory=dict)
    payload: str = ""
    exec_stats: ExecStats = field(default_factory=ExecStats)
    child_agent_id: str | None = None


@dataclass
class Turn:
    """One step of an agent's loop. The unit of the trajectory log."""

    # ── identity ──
    agent_id: str = ""
    agent_type: str = ""
    turn_index: int = 0
    parent_agent_id: str | None = None

    # ── LLM call ──
    llm_input: list[dict] = field(default_factory=list)
    llm_output_text: str = ""
    raw_output: list[dict] = field(default_factory=list)
    parsed_actions: list[dict] = field(default_factory=list)
    llm_stats: LLMStats = field(default_factory=LLMStats)

    # ── observations (one per parsed action) ──
    observations: list[Observation] = field(default_factory=list)
