from __future__ import annotations

try:
    from agents.exceptions import AgentsException as _AgentsException
except ImportError:
    _AgentsException = Exception


class ContextWindowExceeded(_AgentsException):
    """vLLM/OpenAI 400 BadRequestError indicating context overflow.

    Mirrors upstream SWE-agent's ``ContextWindowExceededError`` →
    ``exit_context`` pathway: treat as a normal terminal condition,
    commit whatever the agent has, and submit. ``safe_runner_run``
    re-raises this in place of the raw ``openai.BadRequestError`` so
    callers can distinguish "I ran out of room" from genuine bugs.
    """


class StuckTerminate(_AgentsException):
    pass


class EpisodeStuckAbort(BaseException):
    """Episode-level stuck escalation raised from a delegation tool.

    A per-sub-agent :class:`StuckTerminate` only ends one delegation, so an
    orchestrator can re-delegate the same wedged role indefinitely (observed:
    2546 stuck-terminates in one 10h episode). After a role stuck-terminates
    too many times in a row, the delegate raises THIS to end the whole
    episode. It subclasses ``BaseException`` on purpose: the agents-SDK
    function-tool wrapper catches ``Exception`` and turns tool errors into
    model-visible strings (so a re-raised ``StuckTerminate`` would be
    swallowed and the orchestrator would just continue). ``BaseException``
    bypasses that handler and every ``except Exception`` in the run loop,
    propagating to ``AgentRunner.run`` which finalizes the episode.
    """


class StuckRollback(_AgentsException):

    def __init__(
        self,
        message: str,
        *,
        input_items: list,
        rollback_turns: int,
        rollback_count: int,
        loop_type: str = "?",
        repeats: object = "?",
    ) -> None:
        super().__init__(message)
        self.input_items = input_items
        self.rollback_turns = rollback_turns
        self.rollback_count = rollback_count
        self.loop_type = loop_type
        self.repeats = repeats


def _is_context_overflow_error(exc: Exception) -> bool:
    try:
        from openai import BadRequestError
    except ImportError:
        return False
    if not isinstance(exc, BadRequestError):
        return False
    msg = str(exc).lower()
    return (
        "maximum context length" in msg
        or "context_length_exceeded" in msg
        or ("input tokens" in msg and "max" in msg)
    )
