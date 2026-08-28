"""Stuck detection — direct port of OpenHands ``StuckDetector``.

Upstream reference (commit-pinned):
    third_party/openhands_vendored/stuck.py

The algorithm is identical to upstream (5 scenarios as numbered there).
Only the typed Event/Observation hierarchy is reimplemented locally with
small dataclasses, since OH's class hierarchy isn't available outside the
OH runtime — we run on the openai-agents SDK.

Five scenarios:
  1. last 4 (action, obs) pairs identical
  2. last 3 actions identical AND last 3 obs all errors
  3. last 3 agent text messages identical with no observation between
  4. 6-step alternating pattern: a0=a2=a4 / a1=a3=a5, same for obs
  5. >=10 consecutive AgentCondensationObservation events with no other
     events between (context-window-error loop). Doesn't fire in our
     setup since we removed the condenser, but the scenario is kept for
     fidelity / future-proofing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# special cases in ``_eq_no_pid`` for Cmd/IPython types matching upstream.

@dataclass
class Event:
    pass


@dataclass
class Action(Event):
    pass


@dataclass
class Observation(Event):
    pass


# Upstream sentinel types that we never push but keep for filter parity.
@dataclass
class NullAction(Action):
    pass


@dataclass
class NullObservation(Observation):
    pass


# Source enum analog (upstream ``EventSource``).
class EventSource:
    USER = "user"
    AGENT = "agent"


@dataclass
class MessageAction(Action):
    content: str = ""
    source: str = EventSource.AGENT


@dataclass
class CmdRunAction(Action):
    command: str = ""


@dataclass
class CmdOutputObservation(Observation):
    command: str = ""
    exit_code: int = 0
    content: str = ""


@dataclass
class IPythonRunCellAction(Action):
    """Kept for upstream parity; we don't currently emit this."""
    code: str = ""
    thought: str = ""


@dataclass
class IPythonRunCellObservation(Observation):
    content: str = ""


@dataclass
class ToolCallAction(Action):
    name: str = ""
    args_json: str = ""


@dataclass
class ToolOutputObservation(Observation):
    name: str = ""
    content: str = ""


@dataclass
class ErrorObservation(Observation):
    content: str = ""


@dataclass
class AgentCondensationObservation(Observation):
    content: str = ""


# StuckDetector — direct port of upstream

class StuckDetector:
    SYNTAX_ERROR_MESSAGES = [
        'SyntaxError: unterminated string literal (detected at line',
        'SyntaxError: invalid syntax. Perhaps you forgot a comma?',
        'SyntaxError: incomplete input',
    ]

    @dataclass
    class StuckAnalysis:
        loop_type: str
        loop_repeat_times: int
        loop_start_idx: int

    def __init__(self) -> None:
        self.stuck_analysis: Optional[StuckDetector.StuckAnalysis] = None

    def is_stuck(self, history: list[Event], headless_mode: bool = True) -> bool:
        """Checks if the agent is stuck in a loop.

        Args:
            history: full event history (interleaved Actions + Observations).
            headless_mode: matches upstream — True considers all history,
                False filters to events after the last USER MessageAction.
        """
        filtered_history_offset = 0
        if not headless_mode:
            last_user_msg_idx = -1
            for i, event in enumerate(reversed(history)):
                if (
                    isinstance(event, MessageAction)
                    and event.source == EventSource.USER
                ):
                    last_user_msg_idx = len(history) - i - 1
                    break
            filtered_history_offset = last_user_msg_idx + 1
            history_to_check = history[last_user_msg_idx + 1:]
        else:
            history_to_check = history

        # Filter out user messages and null events (same as upstream).
        filtered_history = [
            event
            for event in history_to_check
            if not (
                (isinstance(event, MessageAction) and event.source == EventSource.USER)
                or isinstance(event, (NullAction, NullObservation))
            )
        ]

        if len(filtered_history) < 3:
            return False

        last_actions: list[Event] = []
        last_observations: list[Event] = []
        for event in reversed(filtered_history):
            if isinstance(event, Action) and len(last_actions) < 4:
                last_actions.append(event)
            elif isinstance(event, Observation) and len(last_observations) < 4:
                last_observations.append(event)
            if len(last_actions) == 4 and len(last_observations) == 4:
                break

        if self._is_stuck_repeating_action_observation(
            last_actions, last_observations, filtered_history, filtered_history_offset
        ):
            return True

        if self._is_stuck_repeating_action_error(
            last_actions, last_observations, filtered_history, filtered_history_offset
        ):
            return True

        if self._is_stuck_monologue(filtered_history, filtered_history_offset):
            return True

        if len(filtered_history) >= 6:
            if self._is_stuck_action_observation_pattern(
                filtered_history, filtered_history_offset
            ):
                return True

        if len(filtered_history) >= 10:
            if self._is_stuck_context_window_error(
                filtered_history, filtered_history_offset
            ):
                return True

        self.stuck_analysis = None
        return False

    def _is_stuck_repeating_action_observation(
        self, last_actions, last_observations, filtered_history, filtered_history_offset=0,
    ) -> bool:
        if len(last_actions) == 4 and len(last_observations) == 4:
            actions_equal = all(
                self._eq_no_pid(last_actions[0], a) for a in last_actions
            )
            observations_equal = all(
                self._eq_no_pid(last_observations[0], o) for o in last_observations
            )
            if actions_equal and observations_equal:
                logger.warning('Action, Observation loop detected')
                self.stuck_analysis = StuckDetector.StuckAnalysis(
                    loop_type='repeating_action_observation',
                    loop_repeat_times=4,
                    loop_start_idx=filtered_history.index(last_actions[-1])
                    + filtered_history_offset,
                )
                return True
        return False

    def _is_stuck_repeating_action_error(
        self, last_actions, last_observations, filtered_history, filtered_history_offset=0,
    ) -> bool:
        if len(last_actions) < 3 or len(last_observations) < 3:
            return False

        if all(self._eq_no_pid(last_actions[0], a) for a in last_actions[:3]):
            if all(isinstance(o, ErrorObservation) for o in last_observations[:3]):
                logger.warning('Action, ErrorObservation loop detected')
                self.stuck_analysis = StuckDetector.StuckAnalysis(
                    loop_type='repeating_action_error',
                    loop_repeat_times=3,
                    loop_start_idx=filtered_history.index(last_actions[-1])
                    + filtered_history_offset,
                )
                return True
            elif all(
                isinstance(o, IPythonRunCellObservation) for o in last_observations[:3]
            ):
                warning = 'Action, IPythonRunCellObservation loop detected'
                for error_message in self.SYNTAX_ERROR_MESSAGES:
                    obs3 = [
                        o for o in last_observations[:3]
                        if isinstance(o, IPythonRunCellObservation)
                    ]
                    if error_message.startswith(
                        'SyntaxError: unterminated string literal (detected at line'
                    ):
                        if self._check_for_consistent_line_error(obs3, error_message):
                            logger.warning(warning)
                            self.stuck_analysis = StuckDetector.StuckAnalysis(
                                loop_type='repeating_action_error',
                                loop_repeat_times=3,
                                loop_start_idx=filtered_history.index(last_actions[-1])
                                + filtered_history_offset,
                            )
                            return True
                    elif error_message in (
                        'SyntaxError: invalid syntax. Perhaps you forgot a comma?',
                        'SyntaxError: incomplete input',
                    ) and self._check_for_consistent_invalid_syntax(obs3, error_message):
                        logger.warning(warning)
                        self.stuck_analysis = StuckDetector.StuckAnalysis(
                            loop_type='repeating_action_error',
                            loop_repeat_times=3,
                            loop_start_idx=filtered_history.index(last_actions[-1])
                            + filtered_history_offset,
                        )
                        return True
        return False

    def _check_for_consistent_invalid_syntax(self, observations, error_message):
        first_lines = []
        valid_observations = []
        for obs in observations:
            content = obs.content
            lines = content.strip().split('\n')
            if len(lines) < 6:
                return False
            line1 = lines[0].strip()
            if not line1.startswith('Cell In[1], line'):
                return False
            first_lines.append(line1)
            if (
                lines[-1].startswith('[Jupyter Python interpreter:')
                and lines[-2].startswith('[Jupyter current working directory:')
                and error_message in lines[-3]
            ):
                valid_observations.append(obs)
        return (
            len(set(first_lines)) == 1
            and len(valid_observations) == 3
            and len(set(
                obs.content.strip().split('\n')[:-2][-1] for obs in valid_observations
            )) == 1
        )

    def _check_for_consistent_line_error(self, observations, error_message):
        error_lines = []
        for obs in observations:
            content = obs.content
            lines = content.strip().split('\n')
            if len(lines) < 3:
                return False
            last_lines = lines[-3:]
            if not (
                last_lines[-2].startswith('[Jupyter current working directory:')
                and last_lines[-1].startswith('[Jupyter Python interpreter:')
            ):
                return False
            if error_message in last_lines[-3]:
                error_lines.append(last_lines[-3])
        return len(error_lines) == 3 and len(set(error_lines)) == 1

    def _is_stuck_monologue(self, filtered_history, filtered_history_offset=0) -> bool:
        agent_message_actions = [
            (i, e) for i, e in enumerate(filtered_history)
            if isinstance(e, MessageAction) and e.source == EventSource.AGENT
        ]
        if len(agent_message_actions) >= 3:
            last3 = agent_message_actions[-3:]
            if all(last3[0][1] == a[1] for a in last3):
                start_index = last3[0][0]
                end_index = last3[-1][0]
                has_observation_between = False
                for event in filtered_history[start_index + 1:end_index]:
                    if isinstance(event, Observation):
                        has_observation_between = True
                        break
                if not has_observation_between:
                    logger.warning('Repeated MessageAction with source=AGENT detected')
                    self.stuck_analysis = StuckDetector.StuckAnalysis(
                        loop_type='monologue',
                        loop_repeat_times=3,
                        loop_start_idx=start_index + filtered_history_offset,
                    )
                    return True
        return False

    def _is_stuck_action_observation_pattern(
        self, filtered_history, filtered_history_offset=0,
    ) -> bool:
        last_six_actions: list[Event] = []
        last_six_observations: list[Event] = []
        for event in reversed(filtered_history):
            if isinstance(event, Action) and len(last_six_actions) < 6:
                last_six_actions.append(event)
            elif isinstance(event, Observation) and len(last_six_observations) < 6:
                last_six_observations.append(event)
            if len(last_six_actions) == 6 and len(last_six_observations) == 6:
                break

        if len(last_six_actions) == 6 and len(last_six_observations) == 6:
            actions_equal = (
                self._eq_no_pid(last_six_actions[0], last_six_actions[2])
                and self._eq_no_pid(last_six_actions[0], last_six_actions[4])
                and self._eq_no_pid(last_six_actions[1], last_six_actions[3])
                and self._eq_no_pid(last_six_actions[1], last_six_actions[5])
            )
            observations_equal = (
                self._eq_no_pid(last_six_observations[0], last_six_observations[2])
                and self._eq_no_pid(last_six_observations[0], last_six_observations[4])
                and self._eq_no_pid(last_six_observations[1], last_six_observations[3])
                and self._eq_no_pid(last_six_observations[1], last_six_observations[5])
            )
            if actions_equal and observations_equal:
                logger.warning('Action, Observation pattern detected')
                self.stuck_analysis = StuckDetector.StuckAnalysis(
                    loop_type='repeating_action_observation_pattern',
                    loop_repeat_times=3,
                    loop_start_idx=filtered_history.index(last_six_actions[-1])
                    + filtered_history_offset,
                )
                return True
        return False

    def _is_stuck_context_window_error(
        self, filtered_history, filtered_history_offset=0,
    ) -> bool:
        condensation_events = [
            (i, e) for i, e in enumerate(filtered_history)
            if isinstance(e, AgentCondensationObservation)
        ]
        if len(condensation_events) < 10:
            return False
        last_condensation_events = condensation_events[-10:]
        for i in range(len(last_condensation_events) - 1):
            start_idx = last_condensation_events[i][0]
            end_idx = last_condensation_events[i + 1][0]
            has_other_events = False
            for event in filtered_history[start_idx + 1:end_idx]:
                if not isinstance(event, AgentCondensationObservation):
                    has_other_events = True
                    break
            if not has_other_events:
                logger.warning(
                    'Context window error loop detected - repeated condensation events'
                )
                self.stuck_analysis = StuckDetector.StuckAnalysis(
                    loop_type='context_window_error',
                    loop_repeat_times=2,
                    loop_start_idx=start_idx + filtered_history_offset,
                )
                return True
        return False

    # ------ equality dispatcher (mirrors upstream _eq_no_pid) -------------
    def _eq_no_pid(self, obj1, obj2) -> bool:
        if isinstance(obj1, IPythonRunCellAction) and isinstance(obj2, IPythonRunCellAction):
            # Match upstream: for edit_file_by_replace, compare first 3 lines of
            if 'edit_file_by_replace(' in obj1.code and 'edit_file_by_replace(' in obj2.code:
                return (
                    len(obj1.code.split('\n')) > 2
                    and obj1.code.split('\n')[:3] == obj2.code.split('\n')[:3]
                )
            return obj1 == obj2
        if isinstance(obj1, CmdOutputObservation) and isinstance(obj2, CmdOutputObservation):
            # Match upstream: ignore ``content`` (PID / output-byte differences
            return obj1.command == obj2.command and obj1.exit_code == obj2.exit_code
        return obj1 == obj2
