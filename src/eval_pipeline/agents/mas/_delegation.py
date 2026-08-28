"""Ephemeral sub-agent delegation tool factory.

Unifies BOAD's ``_delegate`` (issue_analyzer / code_navigator), HyperAgent's
``run_child`` (navigator / editor / executor), and the librarian-overlay
inline delegates into one builder. A fresh ephemeral instance per call
(``CreateMode.NEW``); the registry assigns a unique ``agent_id``. Returns a
plain async ``fn(arg) -> str`` ready to wrap with ``@function_tool``.

Differences across call sites are parameters:
  - ``format_input`` — BOAD wraps the arg as ``invoke_template.format(context=arg)``;
    HyperAgent passes the request through (default identity).
  - ``return_diff`` — HyperAgent's editor appends the ``git diff HEAD`` so
    the planner sees the change.
  - ``text_only_policy`` / ``max_turns`` — forwarded to ``execute_one`` when set.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable

from eval_pipeline.agents.bases.delegation import invoke_agent_logged
from eval_pipeline.agents.llm.profiles import strip_subagent_output
from eval_pipeline.agents.runtime import (
    CreateMode,
    EpisodeStuckAbort,
    StuckTerminate,
)
from eval_pipeline.agents.runtime.agent_registry import AgentBusyError
from eval_pipeline.agents.stuck_detection import (
    ToolCallAction,
    ToolOutputObservation,
)
from eval_pipeline.agents.tools._state import _LAST_TOOL_EVENT_PAIR, _canonical_args

logger = logging.getLogger(__name__)

# Episode-level escalation: a per-sub-agent stuck-terminate only ends that
# one delegation, so an orchestrator that re-delegates the same wedged role
# can loop for the whole run_timeout (observed: one editor stuck-terminated
# 2546× in a single 10h episode). After this many CONSECUTIVE stuck-terminates
# of the same role (reset on any clean return), raise EpisodeStuckAbort so
# the terminate propagates past the orchestrator to the top-level runner,
# which finalizes the episode gracefully (patch extracted,
# agent_error="stuck_terminate"). Override with EVAL_STUCK_BUDGET.
_CONSECUTIVE_STUCK_BUDGET = int(os.environ.get("EVAL_STUCK_BUDGET", "3"))


def make_ephemeral_delegate(
    *,
    mr: Any,
    executor: Any,
    agent: Any,
    agent_role: str,
    max_nudges: int,
    format_input: Callable[[str], str] | None = None,
    return_diff: bool = False,
    docker: Any = None,
    work_dir: str | None = None,
    max_turns: int | None = None,
    text_only_policy: Any = None,
) -> Callable[[str], Awaitable[str]]:
    """Build an ``async fn(arg) -> str`` that runs ``agent`` as an ephemeral
    sub-agent through the registry + logging stack."""

    tool_name = f"delegate_to_{agent_role}"

    def _record(arg: str, content: str) -> str:
        # Publish the delegation as the calling agent's last (Action,
        # Observation) pair so its StuckDetectionHooks see the *delegation*
        # rather than the sub-agent's last bash/editor pair (which leaks via
        # the shared contextvar) or the empty-args fallback in on_tool_end.
        # Both halves are keyed on the REQUEST (not the sub-agent's reply,
        # which is non-deterministic across runs): identical requests yield
        # identical pairs so the repeating-action/observation scenarios fire,
        # while distinct requests yield distinct pairs (no false positive).
        sig = _canonical_args({"request": arg})
        _LAST_TOOL_EVENT_PAIR.set((
            ToolCallAction(name=tool_name, args_json=sig),
            ToolOutputObservation(name=tool_name, content=sig),
        ))
        return content

    async def delegate(arg: str) -> str:
        inp = format_input(arg) if format_input is not None else arg

        async def _execute(inst, hooks):
            kw: dict[str, Any] = {}
            if max_turns is not None:
                kw["max_turns"] = max_turns
            if text_only_policy is not None:
                kw["text_only_policy"] = text_only_policy
            return await executor.execute_one(
                agent=agent, input=inp, hooks=hooks, **kw,
            )

        try:
            result = await invoke_agent_logged(
                **mr.logged(
                    agent_type=agent_role,
                    agent_obj_factory=lambda: agent,
                    execute=_execute,
                    mode=CreateMode.NEW,
                    max_nudges=max_nudges,
                )
            )
            out = strip_subagent_output(result, mr.model)
            if return_diff and docker is not None and work_dir:
                diff_r = await docker.exec_shell(f"cd {work_dir} && git diff HEAD")
                if diff_r.stdout.strip():
                    out += f"\n\n--- git diff ---\n{diff_r.stdout.strip()}"
            # Clean return breaks the consecutive-stuck streak for this role.
            mr.consec_stuck[agent_role] = 0
            return _record(arg, out)
        except AgentBusyError as e:
            msg = f"cannot delegate: {e}"
            logger.warning(msg)
            return _record(arg, msg)
        except StuckTerminate as e:
            n = mr.consec_stuck.get(agent_role, 0) + 1
            mr.consec_stuck[agent_role] = n
            if n >= _CONSECUTIVE_STUCK_BUDGET:
                # The orchestrator keeps re-delegating a wedged role. Escalate
                # to an episode-level abort (BaseException, so the SDK's
                # function-tool error handler can't swallow it) — the
                # top-level runner catches it and finalizes with the current
                # patch instead of burning run_timeout on the same loop.
                logger.error(
                    "%s stuck-terminated %d× in a row — aborting episode",
                    agent_role, n,
                )
                raise EpisodeStuckAbort(
                    f"{agent_role} stuck-terminated {n}× consecutively"
                ) from e
            msg = f"{agent_role} failed: {e}"
            logger.warning("%s (consecutive stuck %d/%d)",
                           msg, n, _CONSECUTIVE_STUCK_BUDGET)
            return _record(arg, msg)
        except Exception as e:  # noqa: BLE001
            msg = f"{agent_role} failed: {e}"
            logger.error(msg)
            return _record(arg, msg)

    return delegate
