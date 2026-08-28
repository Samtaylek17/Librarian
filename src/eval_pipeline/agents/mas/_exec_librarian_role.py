"""Persistent-Executor ("exec-librarian") building blocks.

Execution-role analogue of :mod:`._librarian_role`, reduced to the
persistence mechanism only: one long-lived Executor session per episode
(``UPSERT``), kept with the ``keep_all`` strategy — the librarian's
``<500-new-chars`` pruning measures newly *viewed file content*, which an
executor session (commands, not views) would always fail, silently
disabling persistence. No pointer-only submission channel and no
freshness report; the Executor's tool contract is the verbatim upstream
``bash`` + ``submit_result(result)``.

Deliberately self-contained (small copies instead of new parameters on
:mod:`._librarian_role`) so the modules the running librarian-ablation
experiment imports stay byte-identical.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from eval_pipeline.agents.bases.delegation import invoke_agent_logged
from eval_pipeline.agents.runtime import CreateMode
from eval_pipeline.agents.runtime.agent_registry import AgentBusyError
from eval_pipeline.agents.stuck_detection import (
    ToolCallAction,
    ToolOutputObservation,
)
from eval_pipeline.agents.tools._state import _LAST_TOOL_EVENT_PAIR, _canonical_args

logger = logging.getLogger(__name__)


def make_persistent_executor_call(
    *,
    mr: Any,
    shared_executor: Any,
    executor_id: str,
    caller_role: str,
    max_turns: int,
    max_nudges: int,
    swallow_errors: bool = True,
) -> Callable[[str], Awaitable[str]]:
    """Build the persistent-Executor delegation body ``fn(request) -> str``.

    Resumes the one long-lived Executor instance (``UPSERT``) so its history
    accumulates across calls. Mirrors :func:`._librarian_role.make_librarian_call`
    (query-keyed stuck pair, error handling) with executor wording.
    """

    def _record(request: str, reply: str) -> str:
        # Same request-keyed (Action, Observation) pair as the librarian
        # path — the caller's StuckDetectionHooks flag a caller re-issuing
        # the identical request without collapsing distinct requests.
        sig = _canonical_args({"request": request})
        _LAST_TOOL_EVENT_PAIR.set((
            ToolCallAction(name=executor_id, args_json=sig),
            ToolOutputObservation(name=executor_id, content=sig),
        ))
        return reply

    async def call(request: str) -> str:
        async def _execute(inst, hooks):
            return await shared_executor.invoke(
                request, caller_role=caller_role, hooks=hooks,
                max_turns=max_turns,
            )

        try:
            out = await invoke_agent_logged(
                **mr.logged(
                    agent_type=executor_id,
                    agent_obj_factory=lambda: shared_executor,
                    execute=_execute,
                    agent_id=executor_id,
                    mode=CreateMode.UPSERT,
                    max_nudges=max_nudges,
                )
            )
        except AgentBusyError as e:
            return _record(request, f"cannot delegate to executor: {e}")
        except Exception as e:  # noqa: BLE001
            if not swallow_errors:
                raise
            logger.error("executor failed: %s", e)
            return _record(request, f"executor failed: {e}")

        return _record(request, out)

    return call
