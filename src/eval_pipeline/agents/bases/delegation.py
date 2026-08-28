"""Shared agent-invocation helper: registry lock + per-agent hooks.

Every agent run — top-level or delegated sub-agent — goes through
:func:`invoke_agent_logged`, which is the one place that:

* routes the invocation through the :class:`AgentRegistry` (enforcing the
  non-reentrant lock of requirement 2 and the create-or-resume policy of
  requirement 3, and installing the ``_CURRENT_AGENT`` context that the
  LLM gate of requirement 1 checks), and
* builds the per-agent ``LLMCallLogHooks`` (now tagged with ``agent_type``
  and ``parent_agent_id`` so turns carry full identity + call-tree links)
  plus stuck detection, wrapped in ``_MultiHooks``.

The caller supplies ``execute(instance, hooks)`` — the actual run, which
differs per runner (``executor.execute_one`` for ephemeral sub-agents,
``SharedSubAgent.invoke`` for the persistent librarian). This keeps each
runner's idiosyncratic pre/post-processing in the runner while the
registry + observability wiring lives here once.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from eval_pipeline.agents.bases.openhands_base import StuckDetectionHooks
from eval_pipeline.agents.observability import LLMCallLogHooks
from eval_pipeline.agents.runtime import _MultiHooks
from eval_pipeline.agents.runtime.agent_registry import (
    AgentInstance,
    AgentRegistry,
    CreateMode,
)


async def invoke_agent_logged(
    *,
    registry: AgentRegistry,
    call_log: Any,
    agent_type: str,
    agent_obj_factory: Callable[[], Any],
    execute: Callable[[AgentInstance, Any], Awaitable[Any]],
    model: str,
    tokenizer: Any = None,
    block_size: int = 16,
    energy_url: str | None = None,
    idle_power_W: list[float] | None = None,
    agent_id: str | None = None,
    mode: CreateMode = CreateMode.NEW,
    stuck_hooks_factory: Callable[[], Any] | None = None,
    max_nudges: int = 2,
) -> Any:
    """Resolve + lock the agent, build its hooks, and run ``execute``."""

    async def run_fn(inst: AgentInstance) -> Any:
        log_hooks = LLMCallLogHooks(
            call_log,
            agent_id=inst.agent_id,
            agent_role=agent_type,
            agent_type=agent_type,
            parent_agent_id=inst.parent_agent_id,
            model=model,
            tokenizer=tokenizer,
            block_size=block_size,
            energy_url=energy_url,
            idle_power_W=idle_power_W,
        )
        stuck = (
            stuck_hooks_factory() if stuck_hooks_factory is not None
            else StuckDetectionHooks(max_nudges=max_nudges)
        )
        hooks = _MultiHooks([log_hooks, stuck])
        return await execute(inst, hooks)

    return await registry.invoke_agent(
        agent_type=agent_type,
        agent_id=agent_id,
        mode=mode,
        build_fn=agent_obj_factory,
        run_fn=run_fn,
    )
