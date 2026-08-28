from __future__ import annotations

from eval_pipeline.agents.llm.profiles import (  # noqa: F401
    strip_reasoning,
    strip_subagent_output,
    thinking_extra_body,
)


def disable_thinking_extra_body(model: str | None) -> dict:
    return thinking_extra_body(model, on=False)
