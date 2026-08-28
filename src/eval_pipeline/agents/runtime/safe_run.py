from __future__ import annotations

import asyncio
import logging
from typing import Any

from eval_pipeline.agents.runtime.exceptions import (
    ContextWindowExceeded,
    StuckRollback,
    _is_context_overflow_error,
)
from eval_pipeline.agents.runtime.text_only_policy import (
    TerminatePolicy,
    TextOnlyPolicy,
    _terminated_without_tool_call,
)

logger = logging.getLogger(__name__)


def _item_type(item) -> str | None:
    if isinstance(item, dict):
        return item.get("type")
    return getattr(item, "type", None)


def _truncate_input_by_turns(input_items: list, n_turns: int) -> list:
    n_turns = max(1, int(n_turns))
    boundaries = [
        i for i, item in enumerate(input_items)
        if _item_type(item) == "function_call_output"
    ]
    if len(boundaries) <= n_turns:
        first_action = next(
            (i for i, item in enumerate(input_items)
             if _item_type(item) in ("function_call", "function_call_output")),
            len(input_items),
        )
        return input_items[:first_action]
    keep_until_boundary = boundaries[len(boundaries) - n_turns - 1]
    return input_items[: keep_until_boundary + 1]


def _reset_rollback_hooks(hooks) -> None:
    if hooks is None:
        return
    candidates = [hooks]
    inner = getattr(hooks, "hooks", None)
    if inner is not None:
        candidates = list(inner)
    for h in candidates:
        reset = getattr(h, "_reset_for_rollback", None)
        if callable(reset):
            reset()


async def safe_runner_run(
    agent,
    input_data,
    *,
    text_only_policy: TextOnlyPolicy | None = None,
    **kwargs,
):
    """Run agent with retry on ModelBehaviorError + per-agent text-only policy.

    Catches vLLM's "context length exceeded" 400 and re-raises as
    :class:`ContextWindowExceeded` so all runners can handle it
    uniformly as a graceful terminal condition (matches upstream
    SWE-agent's exit_context behavior). Other ``openai.BadRequestError``
    propagate unchanged.

    ``text_only_policy``: how to handle the case where the LLM ends a
    run by emitting text without any tool_call. Defaults to
    :class:`TerminatePolicy` (= upstream openhands behavior).
    Orchestrator-style runners (BOAD, HyperAgent planner) pass
    :class:`NudgePolicy` to nudge the model into delegating rather
    than dying on the first reflective turn.
    """
    from agents import Runner
    from agents.exceptions import ModelBehaviorError

    if "max_turns" not in kwargs:
        kwargs["max_turns"] = 1000

    max_retries = kwargs.pop("max_model_errors", 3)
    policy = text_only_policy or TerminatePolicy()
    text_only_attempt = 0

    while True:
        did_rollback = False
        for attempt in range(max_retries + 1):
            try:
                try:
                    result = await Runner.run(agent, input_data, **kwargs)
                except Exception as ctx_e:
                    if _is_context_overflow_error(ctx_e):
                        raise ContextWindowExceeded(str(ctx_e)) from ctx_e
                    raise
            except StuckRollback as sr:
                if sr.rollback_turns > 0:
                    truncated = _truncate_input_by_turns(
                        sr.input_items, sr.rollback_turns
                    )
                    nudge = (
                        f"You appear to be stuck in a loop (detector: "
                        f"loop_type={sr.loop_type}, repeats={sr.repeats}). "
                        f"I've removed your last {sr.rollback_turns} turns "
                        f"of conversation. Try a fundamentally different "
                        f"approach — different tool, different file, "
                        f"different framing of the problem. Do not repeat "
                        f"the actions you just took."
                    )
                else:
                    truncated = list(sr.input_items)
                    nudge = (
                        f"You appear to be stuck in a loop (detector: "
                        f"loop_type={sr.loop_type}, repeats={sr.repeats}). "
                        f"Your conversation has been rolled back to the "
                        f"most recent ``branch(...)`` checkpoint you "
                        f"registered. Everything you did since that "
                        f"checkpoint has been discarded. Try a "
                        f"fundamentally different approach for the work "
                        f"that follows the checkpoint — do not repeat "
                        f"the actions you just took."
                    )
                truncated.append({
                    "role": "user",
                    "content": nudge,
                })
                _reset_rollback_hooks(kwargs.get("hooks"))
                logger.warning(
                    "Rolled back %s (rollback %d). New input length=%d.",
                    f"{sr.rollback_turns} turns" if sr.rollback_turns > 0 else "to checkpoint",
                    sr.rollback_count, len(truncated),
                )
                input_data = truncated
                did_rollback = True
                break
            except ModelBehaviorError as e:
                if attempt >= max_retries:
                    raise
                run_data = getattr(e, "run_data", None)
                if run_data is None:
                    raise
                from agents.items import ItemHelpers
                new_input = ItemHelpers.input_to_new_input_list(run_data.input)
                for item in run_data.new_items:
                    try:
                        new_input.append(item.to_input_item())
                    except Exception:
                        pass
                new_input.append({
                    "role": "user",
                    "content": f"Error: {e}. Please use only the tools available to you.",
                })
                input_data = new_input
                continue
            break
        else:
            raise RuntimeError(
                "safe_runner_run: ModelBehaviorError retries exhausted unexpectedly"
            )

        if did_rollback:
            continue

        if not _terminated_without_tool_call(result):
            return result
        action, payload = policy.handle(result, text_only_attempt)
        if action == "terminate":
            return result
        if action == "retry":
            input_data = payload
            text_only_attempt += 1
            continue


async def run_top_agent_with_timeout(
    agent,
    input_data,
    *,
    hooks,
    timeout: float,
    max_turns: int | None = None,
    label: str = "agent",
) -> tuple[Any | None, str | None]:
    kwargs: dict[str, Any] = {"hooks": hooks}
    if max_turns is not None:
        kwargs["max_turns"] = max_turns
    try:
        result = await asyncio.wait_for(
            safe_runner_run(agent, input_data, **kwargs),
            timeout=timeout,
        )
        return result, None
    except asyncio.TimeoutError:
        logger.error("%s timed out after %ds", label, timeout)
        return None, f"run_timeout: exceeded {timeout}s"
    except ContextWindowExceeded as cwe:
        logger.warning("%s context window exceeded: %s", label, cwe)
        return None, f"context_window_exceeded: {cwe}"
