from __future__ import annotations

from eval_pipeline.agents.stuck_detection import (
    ToolCallAction,
    ToolOutputObservation,
)
from eval_pipeline.agents.tools._state import (
    _LAST_TOOL_EVENT_PAIR,
    _canonical_args,
)


def make_oh_submit():
    from agents import function_tool

    @function_tool
    async def submit(result: str = "") -> str:
        text = "Solution submitted."
        _LAST_TOOL_EVENT_PAIR.set((
            ToolCallAction(
                name="submit",
                args_json=_canonical_args({"result": result}),
            ),
            ToolOutputObservation(name="submit", content=text),
        ))
        return text

    return submit
