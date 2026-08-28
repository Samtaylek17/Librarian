"""Shared OH-style harness primitives.

Reusable building blocks composed by every agent module:

- ``_load_oh_system_prompt()`` — loads the vendored OpenHands
  ``system_prompt.j2`` (third_party/openhands_vendored/prompts/, pinned
  via COMMIT_PIN.txt) and returns it as a string. Public entry point
  for ``vanilla.py``'s ``VANILLA_SYSTEM_PROMPT`` constant.
- ``build_openhands_agent(...)`` — constructs an Agents-SDK ``Agent`` with
  the default OH tool set (bash + str_replace_editor + submit) plus any
  ``extra_tools`` and a ``stop_at_tool_names`` hook. Callers override
  ``system_prompt`` to fork an OH base into a sub-agent role.
- ``StuckDetectionHooks`` — RunHook adapter that feeds a
  :class:`StuckDetector` (1:1 port of upstream OH ``stuck.py``, 5
  scenarios). Each sub-agent should get its own instance.
- ``_MultiHooks`` — fan-out RunHook so LLM call logging + stuck detection
  attach side-by-side.

Agent runners themselves live in:
  - ``eval_pipeline.agents.vanilla``        (no MAS overlay)
  - ``eval_pipeline.agents.mas.hyperagent`` (planner + nav/edit/exec)
  - ``eval_pipeline.agents.mas.boad``       (orchestrator + issue_analyzer +
                                             code_navigator + plan-gen)
  - ``eval_pipeline.agents.mas.librarian``  (persistent repo-librarian
                                             sub-agent; reused by the
                                             *_librarian overlays + RAG variants)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from eval_pipeline.agents.openhands import (
    LLMCallLog,
    LLMCallLogHooks,
    _LAST_TOOL_EVENT_PAIR,
    make_tools,
)
from eval_pipeline.agents.runtime import (
    ContextWindowExceeded,
    StuckRollback,
    StuckTerminate,
    _MultiHooks,
    run_top_agent_with_timeout,
    safe_runner_run,
)
from eval_pipeline.agents.stuck_detection import (
    Event,
    MessageAction,
    EventSource,
    StuckDetector,
    ToolCallAction,
    ToolOutputObservation,
)
from eval_pipeline.utils.docker import DockerExecutor

logger = logging.getLogger(__name__)


from eval_pipeline.agents.prompts.oh_system import (  # noqa: F401, E402
    _load_oh_system_prompt,
    load_oh_system_prompt,
)


try:
    from agents import RunHooks as _RunHooksBase
except ImportError:
    _RunHooksBase = object


class StuckDetectionHooks(_RunHooksBase):
    """Stuck detection — uses :class:`StuckDetector` (typed-event port of
    upstream OpenHands ``stuck.py``).

    This class is the agents-SDK <-> typed-event adapter:
      - on_tool_end → read ``_LAST_TOOL_EVENT_PAIR`` (set by each tool
        wrapper) → append (action, observation) to the history.
      - on_llm_end with no tool calls → append :class:`MessageAction`
        (source=AGENT) for monologue detection (scenario 3).
      - After every push, run ``StuckDetector.is_stuck(history)``.

    On detection: terminate the run immediately by raising
    :class:`StuckTerminate`. This matches upstream OpenHands' behavior
    (set status=STUCK and exit the loop with no nudge insertion).
    Earlier versions of this class injected a nudge message into the
    next LLM turn and only terminated after ``max_nudges`` failed
    nudges; ablation showed nudges did not improve outcomes, so the
    nudge path was retired.

    The ``max_nudges`` parameter is accepted but ignored — kept for
    back-compat with existing call sites that still pass it.
    """

    def __init__(self, max_nudges: int = 0) -> None:
        del max_nudges
        self.force_submit = False
        self.history: list[Event] = []
        self.detector = StuckDetector()

    async def on_agent_start(self, context, agent):  # noqa: D401
        pass

    async def on_agent_end(self, context, agent, output):
        pass

    async def on_handoff(self, context, from_agent, to_agent):
        pass

    async def on_tool_start(self, context, agent, tool):
        pass

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        # No nudge injection — match upstream OpenHands semantics.
        pass

    async def on_llm_end(self, context, agent, response):
        has_tool_call = False
        text = ""
        if hasattr(response, "output"):
            for item in response.output:
                if hasattr(item, "name"):
                    has_tool_call = True
                elif hasattr(item, "content") and isinstance(item.content, str):
                    text += item.content
        if has_tool_call:
            return
        text = text.strip()
        if not text:
            return
        self.history.append(MessageAction(content=text, source=EventSource.AGENT))
        self._check_stuck()

    async def on_tool_end(self, context, agent, tool, result):
        pair = _LAST_TOOL_EVENT_PAIR.get()
        _LAST_TOOL_EVENT_PAIR.set(None)

        if pair is not None:
            action, observation = pair
        else:
            name = getattr(tool, "name", str(tool))
            action = ToolCallAction(name=name, args_json="")
            observation = ToolOutputObservation(name=name, content=str(result))

        self.history.append(action)
        self.history.append(observation)
        self._check_stuck()

    def _check_stuck(self) -> None:
        if not self.detector.is_stuck(self.history, headless_mode=True):
            return
        analysis = self.detector.stuck_analysis
        loop_type = analysis.loop_type if analysis else "?"
        repeats = analysis.loop_repeat_times if analysis else "?"
        logger.warning(
            "Stuck detected (loop_type=%s repeats=%s) — terminating run",
            loop_type, repeats,
        )
        self.force_submit = True
        # Match upstream OpenHands: detect → set status=STUCK and exit.
        raise StuckTerminate(
            f"stuck_terminate (loop: {loop_type}, repeats: {repeats})"
        )


class StuckRollbackHooks(StuckDetectionHooks):

    def __init__(
        self,
        rollback_turns: int = 10,
        max_rollbacks: int = 3,
        max_nudges: int = 0,
    ) -> None:
        super().__init__(max_nudges=max_nudges)
        self.rollback_turns = int(rollback_turns)
        self.max_rollbacks = int(max_rollbacks)
        self.rollback_count = 0
        self._last_input_items: list = []

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        self._last_input_items = list(input_items)
        await super().on_llm_start(context, agent, system_prompt, input_items)

    def _reset_for_rollback(self) -> None:
        self.history = []
        self.detector = StuckDetector()
        self.force_submit = False

    def _check_stuck(self) -> None:
        if not self.detector.is_stuck(self.history, headless_mode=True):
            return
        analysis = self.detector.stuck_analysis
        loop_type = analysis.loop_type if analysis else "?"
        repeats = analysis.loop_repeat_times if analysis else "?"

        if self.rollback_count >= self.max_rollbacks:
            logger.warning(
                "Stuck detected (loop_type=%s repeats=%s) — max_rollbacks "
                "(%d) reached, terminating run",
                loop_type, repeats, self.max_rollbacks,
            )
            self.force_submit = True
            raise StuckTerminate(
                f"stuck_terminate (loop: {loop_type}, repeats: {repeats}, "
                f"after {self.rollback_count} rollbacks)"
            )

        self.rollback_count += 1
        logger.warning(
            "Stuck detected (loop_type=%s repeats=%s) — rolling back "
            "%d turns (rollback %d/%d)",
            loop_type, repeats,
            self.rollback_turns, self.rollback_count, self.max_rollbacks,
        )
        raise StuckRollback(
            f"stuck_rollback (loop: {loop_type}, count: {self.rollback_count})",
            input_items=self._last_input_items,
            rollback_turns=self.rollback_turns,
            rollback_count=self.rollback_count,
            loop_type=loop_type,
            repeats=repeats,
        )


from eval_pipeline.agents.factory import (  # noqa: F401, E402
    build_agent,
    build_openhands_agent,
)
