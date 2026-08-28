"""Phase 3: turns are built per LLM call and observations are bound
synchronously on tool end, with per-execution statistics recovered from
the typed tool event pair."""

from __future__ import annotations

from types import SimpleNamespace

from eval_pipeline.agents.observability.call_log import LLMCallLog
from eval_pipeline.agents.observability.hooks import LLMCallLogHooks
from eval_pipeline.agents.runtime import agent_registry as ar
from eval_pipeline.agents.stuck_detection import (
    CmdOutputObservation,
    CmdRunAction,
)
from eval_pipeline.agents.tools._state import _LAST_TOOL_EVENT_PAIR


def _usage(inp, out, cached):
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
    )


def _resp(output, usage):
    return SimpleNamespace(output=output, usage=usage)


def _tool(name):
    return SimpleNamespace(name=name)


async def _emit_turn(hooks, *, output, usage):
    await hooks.on_llm_start(None, None, "SYS", [{"role": "user", "content": "x"}])
    await hooks.on_llm_end(None, None, _resp(output, usage))


async def test_turn_records_identity_and_stats():
    log = LLMCallLog()
    hooks = LLMCallLogHooks(log, agent_id="a", agent_role="agent",
                            agent_type="vanilla")
    await _emit_turn(
        hooks,
        output=[
            {"type": "message", "content": "thinking"},
            {"type": "function_call", "name": "bash",
             "arguments": '{"command": "ls"}'},
        ],
        usage=_usage(100, 20, 8),
    )
    assert len(log.turns) == 1
    t = log.turns[0]
    assert t.agent_id == "a"
    assert t.agent_type == "vanilla"
    assert t.turn_index == 0
    assert t.parent_agent_id is None
    assert t.llm_output_text == "thinking"
    assert len(t.parsed_actions) == 1
    assert t.llm_stats.input_tokens == 100
    assert t.llm_stats.output_tokens == 20
    assert t.llm_stats.cached_tokens == 8
    assert t.observations == []


async def test_observation_bound_synchronously_with_exec_stats():
    log = LLMCallLog()
    hooks = LLMCallLogHooks(log, agent_id="a", agent_role="agent")
    await _emit_turn(
        hooks,
        output=[{"type": "function_call", "name": "bash", "arguments": "{}"}],
        usage=_usage(10, 5, 0),
    )
    t = log.turns[0]

    full = "line\n" * 1000          # untruncated content the wrapper stashes
    shown = "line\nline\n"          # truncated string the model saw
    await hooks.on_tool_start(None, None, _tool("bash"))
    _LAST_TOOL_EVENT_PAIR.set((
        CmdRunAction(command="ls -la"),
        CmdOutputObservation(command="ls -la", exit_code=0, content=full),
    ))
    await hooks.on_tool_end(None, None, _tool("bash"), shown)

    assert len(t.observations) == 1            # bound to the SAME open turn
    o = t.observations[0]
    assert o.tool_name == "bash"
    assert o.args == {"command": "ls -la"}
    assert o.payload == full                    # verbatim, untruncated
    assert o.exec_stats.exit_code == 0
    assert o.exec_stats.truncated is True       # full longer than shown
    assert o.exec_stats.output_len == len(full)
    assert o.child_agent_id is None


async def test_no_stale_child_link_on_plain_tool(monkeypatch):
    log = LLMCallLog()
    hooks = LLMCallLogHooks(log, agent_id="a", agent_role="agent")
    await _emit_turn(
        hooks,
        output=[{"type": "function_call", "name": "bash", "arguments": "{}"}],
        usage=_usage(10, 5, 0),
    )
    t = log.turns[0]
    # A previous delegation left a child id set...
    ar._LAST_CHILD_AGENT_ID.set("navigator-abc")
    await hooks.on_tool_start(None, None, _tool("bash"))  # clears it
    _LAST_TOOL_EVENT_PAIR.set((
        CmdRunAction(command="echo hi"),
        CmdOutputObservation(command="echo hi", exit_code=0, content="hi"),
    ))
    await hooks.on_tool_end(None, None, _tool("bash"), "hi")
    assert t.observations[0].child_agent_id is None      # not inherited


async def test_child_link_recorded_for_agent_call():
    log = LLMCallLog()
    hooks = LLMCallLogHooks(log, agent_id="root", agent_role="agent")
    await _emit_turn(
        hooks,
        output=[{"type": "function_call", "name": "delegate_to_navigator",
                 "arguments": "{}"}],
        usage=_usage(10, 5, 0),
    )
    t = log.turns[0]
    await hooks.on_tool_start(None, None, _tool("delegate_to_navigator"))
    # Simulate invoke_agent setting the child link during tool execution.
    ar._LAST_CHILD_AGENT_ID.set("navigator-xyz")
    _LAST_TOOL_EVENT_PAIR.set(None)
    await hooks.on_tool_end(None, None, _tool("delegate_to_navigator"),
                            "child result")
    o = t.observations[0]
    assert o.child_agent_id == "navigator-xyz"
    assert o.payload == "child result"


async def test_multiple_turns_increment_index():
    log = LLMCallLog()
    hooks = LLMCallLogHooks(log, agent_id="a", agent_role="agent")
    for _ in range(3):
        await _emit_turn(
            hooks,
            output=[{"type": "function_call", "name": "bash",
                     "arguments": "{}"}],
            usage=_usage(1, 1, 0),
        )
    assert [t.turn_index for t in log.turns] == [0, 1, 2]
    assert log.total() == {"input_tokens": 3, "output_tokens": 3,
                           "cached_tokens": 0}
