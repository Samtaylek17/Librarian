"""Per-agent policy for handling text-only LLM responses.

Different agent roles need different responses to a turn that emitted
message content but no tool_call:

  - file_agent / vanilla single-agent: text IS the answer. Terminate
    and return ``result.final_output``. Matches upstream openhands
    ``MessageAction(wait_for_response=True) → STOPPED`` in autonomous mode.

  - orchestrator (BOAD, HyperAgent planner, etc.): text-only on the
    first turn means "I thought but didn't act". A user-nudge retry is
    needed; without it the orchestrator dies before delegating to any
    sub-agent.

A policy is a small object passed to ``safe_runner_run`` via
``text_only_policy=...``. Defaults to :class:`TerminatePolicy`
(matches upstream OH).
"""

from __future__ import annotations

from typing import Any


class TextOnlyPolicy:

    def handle(self, result, attempt_n: int):
        return ("terminate", None)


class TerminatePolicy(TextOnlyPolicy):
    pass


class NudgePolicy(TextOnlyPolicy):

    DEFAULT_MESSAGE = (
        "Your previous response had no tool call. Please take an action "
        "by calling one of the available tools — do not just describe "
        "what you would do."
    )

    def __init__(self, max_retries: int = 2, message: str | None = None) -> None:
        self.max_retries = max_retries
        self.message = message or self.DEFAULT_MESSAGE

    def handle(self, result, attempt_n: int):
        if attempt_n >= self.max_retries:
            return ("terminate", None)
        from agents.items import ItemHelpers
        new_input = ItemHelpers.input_to_new_input_list(result.input)
        for item in result.new_items:
            try:
                new_input.append(item.to_input_item())
            except Exception:
                pass
        new_input.append({"role": "user", "content": self.message})
        return ("retry", new_input)


def _terminated_without_tool_call(result: Any) -> bool:
    new_items = getattr(result, "new_items", []) or []
    if not new_items:
        return False
    for item in reversed(new_items):
        name = type(item).__name__
        if "ToolCall" in name and "Output" not in name:
            return False
        if "MessageOutputItem" in name or "ReasoningItem" in name:
            return True
    return False
