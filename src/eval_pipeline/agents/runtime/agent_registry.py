"""Central agent runtime: registry + lifecycle + invocation policy.

One ``AgentRegistry`` per episode owns every agent instance and is the
single funnel through which agents are created and invoked. It enforces
three invariants demanded by the refactor:

1. **Only agents call the LLM.** The registry sets the ``_CURRENT_AGENT``
   contextvar around every invocation; the LLM client wrapper (see
   ``llm/vllm.py``) refuses to fire when that context is empty. The
   registry is the only thing that sets it.

2. **No re-entrant invocation of a busy agent.** Each instance carries an
   :class:`AgentState`. ``invoke_agent`` acquires the target as a
   *non-reentrant lock*: an instance that is ``RUNNING`` or ``WAITING``
   cannot be invoked. Because execution is a synchronous ``await`` chain,
   the set of ``WAITING`` instances is exactly the ancestor chain — so
   this single rule rejects both duplicate concurrent calls *and* cycles
   (A→B→A would find A ``WAITING``), eliminating deadlock at the source.

3. **Explicit create-or-resume policy.** ``invoke_agent`` takes a
   :class:`CreateMode`:

     ============  ==========================================  ===========
     caller form   mode                                        on conflict
     ============  ==========================================  ===========
     id omitted    ``NEW``    random id, always fresh instance  (n/a)
     id given (i)  ``CREATE`` must NOT already exist            error
     id given (ii) ``RESUME`` must already exist                error
     id given(iii) ``UPSERT`` resume if present else create     (none)
     ============  ==========================================  ===========

The registry is deliberately LLM-agnostic: building the underlying agent
object and running one invocation are injected as ``build_fn`` / ``run_fn``
callables, so the lifecycle logic is unit-testable without a model.
"""

from __future__ import annotations

import contextvars
import logging
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# The agent_id currently executing (caller, from the LLM's point of view).
# Set ONLY by AgentRegistry.invoke_agent. Read by the LLM client wrapper to
# enforce "only agents call the LLM", and by invoke_agent to link a child to
# its parent.
_CURRENT_AGENT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_CURRENT_AGENT", default=None,
)

# The agent_id of the most recently completed invocation. Set by
# invoke_agent on return so a parent's agent-call tool observation can link
# to the child it spawned (Turn → Observation.child_agent_id).
_LAST_CHILD_AGENT_ID: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("_LAST_CHILD_AGENT_ID", default=None)
)


def current_agent_id() -> str | None:
    """The agent_id on whose behalf code is currently running, or None."""
    return _CURRENT_AGENT.get()


def last_child_agent_id() -> str | None:
    """The agent_id of the most recently completed child invocation."""
    return _LAST_CHILD_AGENT_ID.get()


def clear_last_child_agent_id() -> None:
    """Reset the child link — called before each tool runs so a non
    agent-call tool does not inherit a stale child id."""
    _LAST_CHILD_AGENT_ID.set(None)


class NonAgentLLMCallError(RuntimeError):
    """An LLM request was issued outside any agent context."""


_GATE_WARNED = False


def enforce_llm_agent_gate() -> None:
    """Guardrail for "only agents call the LLM" (requirement 1).

    Called from the LLM client wrapper before every request. Behaviour is
    controlled by ``EVAL_LLM_AGENT_GATE``:

      ``off``    — disabled.
      ``warn``   — log once if a call happens with no current agent
                   (default; surfaces a violation without breaking a run).
      ``strict`` — raise :class:`NonAgentLLMCallError` (hard enforcement of
                   requirement 1; recommended once all runners are
                   validated to route every LLM call through the registry).

    A call is "from an agent" iff ``_CURRENT_AGENT`` is set, which only
    :func:`AgentRegistry.invoke_agent` does.
    """
    mode = os.environ.get("EVAL_LLM_AGENT_GATE", "warn").lower()
    if mode == "off" or current_agent_id() is not None:
        return
    msg = (
        "LLM request issued outside any agent context — only agents may "
        "call the LLM (enter via AgentRegistry.invoke_agent)"
    )
    if mode == "strict":
        raise NonAgentLLMCallError(msg)
    global _GATE_WARNED
    if not _GATE_WARNED:
        _GATE_WARNED = True
        logger.warning("%s [EVAL_LLM_AGENT_GATE=warn]", msg)


class AgentState(Enum):
    IDLE = "idle"        # created or finished a call; invokable
    RUNNING = "running"  # actively executing its own loop
    WAITING = "waiting"  # blocked on an agent-call tool (awaiting a child)
    DONE = "done"        # terminal; not reused (reserved, unused for now)


class CreateMode(Enum):
    NEW = "new"        # no id → random id + fresh instance
    CREATE = "create"  # id given; error if it already exists
    RESUME = "resume"  # id given; error if it does not exist
    UPSERT = "upsert"  # id given; resume if present else create


class AgentBusyError(RuntimeError):
    """Target agent is mid-invocation (RUNNING/WAITING) — cannot be
    invoked. Surfaced to the calling LLM as an observation, not a crash."""

    def __init__(self, agent_id: str, state: "AgentState") -> None:
        super().__init__(
            f"agent '{agent_id}' is busy (state={state.value}); "
            "cannot invoke an agent that is awaiting a response"
        )
        self.agent_id = agent_id
        self.state = state


class AgentExistsError(RuntimeError):
    def __init__(self, agent_id: str) -> None:
        super().__init__(f"agent '{agent_id}' already exists (mode=CREATE)")
        self.agent_id = agent_id


class AgentMissingError(RuntimeError):
    def __init__(self, agent_id: str) -> None:
        super().__init__(f"agent '{agent_id}' does not exist (mode=RESUME)")
        self.agent_id = agent_id


@dataclass
class AgentInstance:
    """A live agent: its identity, lifecycle state, built agent object,
    persistent history, and the turns it has produced."""

    agent_id: str
    agent_type: str
    state: AgentState = AgentState.IDLE
    agent_obj: Any = None
    parent_agent_id: str | None = None
    history: list = field(default_factory=list)
    turns: list = field(default_factory=list)
    n_invocations: int = 0


class AgentRegistry:

    def __init__(
        self,
        *,
        id_factory: Callable[[str], str] | None = None,
    ) -> None:
        self._agents: dict[str, AgentInstance] = {}
        # Injectable for deterministic tests; defaults to type-prefixed uuid.
        self._id_factory = id_factory or _default_id

    # ── inspection ──────────────────────────────────────────────────
    def get(self, agent_id: str) -> AgentInstance | None:
        return self._agents.get(agent_id)

    def exists(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def all(self) -> list[AgentInstance]:
        return list(self._agents.values())

    # ── creation policy ─────────────────────────────────────────────
    def _new_id(self, agent_type: str) -> str:
        while True:
            aid = self._id_factory(agent_type)
            if aid not in self._agents:
                return aid

    def _create(
        self, agent_id: str, agent_type: str, build_fn: Callable[[], Any],
    ) -> AgentInstance:
        inst = AgentInstance(
            agent_id=agent_id,
            agent_type=agent_type,
            agent_obj=build_fn() if build_fn is not None else None,
        )
        self._agents[agent_id] = inst
        return inst

    def resolve(
        self,
        *,
        agent_type: str,
        agent_id: str | None,
        mode: CreateMode,
        build_fn: Callable[[], Any] | None = None,
    ) -> AgentInstance:
        """Apply the create-or-resume policy and return the instance.

        Does NOT change state or run anything — see :meth:`invoke_agent`.
        """
        if mode is CreateMode.NEW:
            if agent_id is not None:
                raise ValueError("mode=NEW must not be given an agent_id")
            return self._create(self._new_id(agent_type), agent_type, build_fn)

        if agent_id is None:
            raise ValueError(f"mode={mode.value} requires an agent_id")

        if mode is CreateMode.CREATE:
            if agent_id in self._agents:
                raise AgentExistsError(agent_id)
            return self._create(agent_id, agent_type, build_fn)

        if mode is CreateMode.RESUME:
            inst = self._agents.get(agent_id)
            if inst is None:
                raise AgentMissingError(agent_id)
            return inst

        if mode is CreateMode.UPSERT:
            inst = self._agents.get(agent_id)
            if inst is None:
                inst = self._create(agent_id, agent_type, build_fn)
            return inst

        raise ValueError(f"unknown CreateMode: {mode!r}")  # pragma: no cover

    # ── invocation (the non-reentrant lock) ─────────────────────────
    async def invoke_agent(
        self,
        *,
        agent_type: str,
        run_fn: Callable[[AgentInstance], Awaitable[Any]],
        agent_id: str | None = None,
        mode: CreateMode = CreateMode.NEW,
        build_fn: Callable[[], Any] | None = None,
    ) -> Any:
        """Resolve, acquire as a non-reentrant lock, run, release.

        ``run_fn(instance)`` performs the actual agent loop (the registry
        stays LLM-agnostic). The current agent is marked ``WAITING`` for
        the duration; the target is marked ``RUNNING`` and installed as the
        new ``_CURRENT_AGENT`` so nested invocations and LLM calls see the
        correct caller.
        """
        inst = self.resolve(
            agent_type=agent_type, agent_id=agent_id,
            mode=mode, build_fn=build_fn,
        )

        if inst.state in (AgentState.RUNNING, AgentState.WAITING):
            raise AgentBusyError(inst.agent_id, inst.state)

        parent_id = _CURRENT_AGENT.get()
        parent = self._agents.get(parent_id) if parent_id else None
        inst.parent_agent_id = parent_id

        if parent is not None:
            parent.state = AgentState.WAITING
        inst.state = AgentState.RUNNING
        inst.n_invocations += 1
        token = _CURRENT_AGENT.set(inst.agent_id)
        try:
            return await run_fn(inst)
        finally:
            _CURRENT_AGENT.reset(token)
            _LAST_CHILD_AGENT_ID.set(inst.agent_id)
            inst.state = AgentState.IDLE
            if parent is not None:
                parent.state = AgentState.RUNNING


def _default_id(agent_type: str) -> str:
    return f"{agent_type}-{uuid.uuid4().hex[:6]}"
