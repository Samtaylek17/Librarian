"""Unit tests for the agent runtime registry (Phase 1).

Pure lifecycle logic — no LLM, no Docker. ``run_fn`` / ``build_fn`` are
injected fakes. Covers the create-or-resume policy and the
non-reentrant-lock invariant (busy rejection + cycle rejection).
"""

from __future__ import annotations

import itertools

import pytest

from eval_pipeline.agents.runtime import agent_registry as ar
from eval_pipeline.agents.runtime.agent_registry import (
    AgentBusyError,
    AgentExistsError,
    AgentMissingError,
    AgentRegistry,
    AgentState,
    CreateMode,
    NonAgentLLMCallError,
    current_agent_id,
    enforce_llm_agent_gate,
)


def _counter_factory():
    """Deterministic id factory: '<type>-0', '<type>-1', ..."""
    counters: dict[str, itertools.count] = {}

    def make(agent_type: str) -> str:
        c = counters.setdefault(agent_type, itertools.count())
        return f"{agent_type}-{next(c)}"

    return make


def _noop_build():
    return object()


async def _return(value):
    async def run(_inst):
        return value
    return run


# ── create-or-resume policy ─────────────────────────────────────────────

def test_new_generates_unique_ids():
    reg = AgentRegistry(id_factory=_counter_factory())
    a = reg.resolve(agent_type="nav", agent_id=None, mode=CreateMode.NEW,
                    build_fn=_noop_build)
    b = reg.resolve(agent_type="nav", agent_id=None, mode=CreateMode.NEW,
                    build_fn=_noop_build)
    assert a.agent_id != b.agent_id
    assert a.agent_id == "nav-0" and b.agent_id == "nav-1"
    assert reg.exists("nav-0") and reg.exists("nav-1")


def test_new_rejects_explicit_id():
    reg = AgentRegistry(id_factory=_counter_factory())
    with pytest.raises(ValueError):
        reg.resolve(agent_type="nav", agent_id="nav-x", mode=CreateMode.NEW,
                    build_fn=_noop_build)


def test_create_errors_if_exists():
    reg = AgentRegistry(id_factory=_counter_factory())
    reg.resolve(agent_type="lib", agent_id="lib", mode=CreateMode.CREATE,
                build_fn=_noop_build)
    with pytest.raises(AgentExistsError):
        reg.resolve(agent_type="lib", agent_id="lib", mode=CreateMode.CREATE,
                    build_fn=_noop_build)


def test_resume_errors_if_missing_then_reuses():
    reg = AgentRegistry(id_factory=_counter_factory())
    with pytest.raises(AgentMissingError):
        reg.resolve(agent_type="lib", agent_id="lib", mode=CreateMode.RESUME)
    created = reg.resolve(agent_type="lib", agent_id="lib",
                          mode=CreateMode.CREATE, build_fn=_noop_build)
    resumed = reg.resolve(agent_type="lib", agent_id="lib",
                          mode=CreateMode.RESUME)
    assert resumed is created


def test_upsert_creates_then_resumes():
    reg = AgentRegistry(id_factory=_counter_factory())
    first = reg.resolve(agent_type="lib", agent_id="lib",
                        mode=CreateMode.UPSERT, build_fn=_noop_build)
    second = reg.resolve(agent_type="lib", agent_id="lib",
                         mode=CreateMode.UPSERT, build_fn=_noop_build)
    assert first is second


def test_resume_requires_id():
    reg = AgentRegistry(id_factory=_counter_factory())
    with pytest.raises(ValueError):
        reg.resolve(agent_type="lib", agent_id=None, mode=CreateMode.RESUME)


# ── invocation + non-reentrant lock ─────────────────────────────────────

async def test_invoke_runs_and_returns_to_idle():
    reg = AgentRegistry(id_factory=_counter_factory())
    seen = {}

    async def run(inst):
        seen["state"] = inst.state
        seen["current"] = current_agent_id()
        return "ok"

    out = await reg.invoke_agent(agent_type="nav", mode=CreateMode.NEW,
                                 build_fn=_noop_build, run_fn=run)
    assert out == "ok"
    assert seen["state"] is AgentState.RUNNING
    assert seen["current"] == "nav-0"
    assert reg.get("nav-0").state is AgentState.IDLE
    assert current_agent_id() is None  # contextvar restored
    assert reg.get("nav-0").n_invocations == 1


async def test_self_reentry_is_rejected():
    reg = AgentRegistry(id_factory=_counter_factory())

    async def run(inst):
        # While 'a' is RUNNING, invoking it again must be rejected.
        with pytest.raises(AgentBusyError):
            await reg.invoke_agent(agent_type="a", agent_id="a",
                                   mode=CreateMode.RESUME,
                                   run_fn=(lambda i: None))
        return "done"

    out = await reg.invoke_agent(agent_type="a", agent_id="a",
                                 mode=CreateMode.CREATE, build_fn=_noop_build,
                                 run_fn=run)
    assert out == "done"


async def test_cycle_is_rejected():
    reg = AgentRegistry(id_factory=_counter_factory())
    order = []

    async def run_b(inst):
        order.append("b")
        # A is WAITING (it is our ancestor) → cannot be invoked.
        with pytest.raises(AgentBusyError) as ei:
            await reg.invoke_agent(agent_type="a", agent_id="a",
                                   mode=CreateMode.RESUME,
                                   run_fn=(lambda i: None))
        assert ei.value.state is AgentState.WAITING
        return "b-done"

    async def run_a(inst):
        order.append("a")
        assert reg.get("a").state is AgentState.RUNNING
        res = await reg.invoke_agent(agent_type="b", agent_id="b",
                                     mode=CreateMode.CREATE,
                                     build_fn=_noop_build, run_fn=run_b)
        # After child returns, A is RUNNING again, B is IDLE.
        assert reg.get("a").state is AgentState.RUNNING
        assert reg.get("b").state is AgentState.IDLE
        return res

    out = await reg.invoke_agent(agent_type="a", agent_id="a",
                                 mode=CreateMode.CREATE, build_fn=_noop_build,
                                 run_fn=run_a)
    assert out == "b-done"
    assert order == ["a", "b"]
    assert reg.get("a").state is AgentState.IDLE
    assert reg.get("b").state is AgentState.IDLE


async def test_parent_child_linkage():
    reg = AgentRegistry(id_factory=_counter_factory())

    async def run_child(inst):
        assert inst.parent_agent_id == "root"
        return "child"

    async def run_root(inst):
        assert inst.parent_agent_id is None
        return await reg.invoke_agent(agent_type="child", mode=CreateMode.NEW,
                                      build_fn=_noop_build, run_fn=run_child)

    await reg.invoke_agent(agent_type="root", agent_id="root",
                           mode=CreateMode.CREATE, build_fn=_noop_build,
                           run_fn=run_root)
    assert reg.get("child-0").parent_agent_id == "root"


async def test_persistent_agent_reused_between_calls():
    """A finished (IDLE) agent can be RESUMEd again — the librarian
    pattern. Busy rejection only applies to in-flight agents."""
    reg = AgentRegistry(id_factory=_counter_factory())

    async def run(inst):
        return inst.n_invocations

    first = await reg.invoke_agent(agent_type="lib", agent_id="lib",
                                   mode=CreateMode.UPSERT, build_fn=_noop_build,
                                   run_fn=run)
    second = await reg.invoke_agent(agent_type="lib", agent_id="lib",
                                    mode=CreateMode.RESUME, run_fn=run)
    assert first == 1 and second == 2
    assert reg.get("lib").state is AgentState.IDLE


# ── LLM access gate (requirement 1) ─────────────────────────────────────

def test_gate_off_is_noop(monkeypatch):
    monkeypatch.setenv("EVAL_LLM_AGENT_GATE", "off")
    enforce_llm_agent_gate()  # no current agent, but off → no raise


def test_gate_strict_raises_without_agent(monkeypatch):
    monkeypatch.setenv("EVAL_LLM_AGENT_GATE", "strict")
    with pytest.raises(NonAgentLLMCallError):
        enforce_llm_agent_gate()


def test_gate_warn_does_not_raise(monkeypatch):
    monkeypatch.setenv("EVAL_LLM_AGENT_GATE", "warn")
    monkeypatch.setattr(ar, "_GATE_WARNED", False)
    enforce_llm_agent_gate()  # warns once, never raises


async def test_gate_strict_passes_inside_agent(monkeypatch):
    monkeypatch.setenv("EVAL_LLM_AGENT_GATE", "strict")
    reg = AgentRegistry(id_factory=_counter_factory())

    async def run(inst):
        enforce_llm_agent_gate()  # current agent set → no raise
        return "ok"

    out = await reg.invoke_agent(agent_type="a", mode=CreateMode.NEW,
                                 build_fn=_noop_build, run_fn=run)
    assert out == "ok"


async def test_state_restored_on_exception():
    reg = AgentRegistry(id_factory=_counter_factory())

    async def boom(inst):
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        await reg.invoke_agent(agent_type="a", agent_id="a",
                               mode=CreateMode.CREATE, build_fn=_noop_build,
                               run_fn=boom)
    # lock released, contextvar restored even though run_fn threw
    assert reg.get("a").state is AgentState.IDLE
    assert current_agent_id() is None
