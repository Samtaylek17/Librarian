"""Reusable Repo-Librarian building blocks.

The persistent librarian sub-agent is shared by ``librarian`` and the two
``*_librarian`` overlays (and their RAG subclasses). Three pieces used to be
copy-pasted into each:

  - :func:`make_view_command_submit_tool` — the ``submit_result(result,
    view_commands)`` side-channel: the librarian returns ``[path, start,
    end]`` triples; the system runs them server-side (``awk`` cat -n) and
    prepends the output to what the caller sees, keeping the librarian's own
    token budget lean. Byte-identical in all three call sites before.

  - :class:`LibrarianRoleMixin` — ``_make_navigator`` (wrap the librarian
    agent in a persistent :class:`SharedSubAgent`) plus the memory-strategy
    constructor handling. RAG subclasses override ``_make_navigator``.

  - :func:`make_librarian_call` — the delegation body: clear the view
    buffer, invoke the persistent librarian (``UPSERT`` so its history
    accumulates), prepend any view output. Each runner wraps this in its own
    ``@function_tool`` so the LLM-facing docstring stays verbatim.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from agents import function_tool

from eval_pipeline.agents.bases.delegation import invoke_agent_logged
from eval_pipeline.agents.history import SharedSubAgent
from eval_pipeline.agents.llm.profiles import strip_subagent_output
from eval_pipeline.agents.history.strategies import build_memory_strategy
from eval_pipeline.agents.runtime import CreateMode
from eval_pipeline.agents.runtime.agent_registry import AgentBusyError
from eval_pipeline.agents.stuck_detection import (
    ToolCallAction,
    ToolOutputObservation,
)
from eval_pipeline.agents.tools._state import _LAST_TOOL_EVENT_PAIR, _canonical_args

logger = logging.getLogger(__name__)


def make_view_command_submit_tool(docker: Any, pending_view_output: list[str]):
    """Build the librarian's ``submit_result(result, view_commands)`` tool.

    ``view_commands`` is a JSON list of ``[path, start, end]`` triples; each
    is rendered server-side (cat -n via ``awk``) into ``pending_view_output``
    so the caller can prepend it. The view does NOT enter the librarian's own
    conversation history.
    """

    @function_tool(name_override="submit_result")
    async def submit_result(result: str, view_commands: str | None = None) -> str:
        outputs: list[str] = []
        if view_commands:
            try:
                cmds = json.loads(view_commands)
                if not isinstance(cmds, list):
                    raise ValueError("view_commands must be a JSON list")
            except Exception as e:  # noqa: BLE001
                pending_view_output.clear()
                return (
                    f"[submit_result error: view_commands not parseable as "
                    f"JSON list of [path, start, end] — {e}]\n\n{result}"
                )
            for entry in cmds:
                if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                    outputs.append(f"[skipped malformed entry: {entry!r}]")
                    continue
                path, start, end = entry
                try:
                    start = int(start)
                    end = int(end)
                except Exception:  # noqa: BLE001
                    outputs.append(f"[skipped: non-integer line range in {entry!r}]")
                    continue
                r = await docker.exec_shell(
                    f"awk 'NR>={start} && NR<={end} "
                    f'{{printf "%6d\\t%s\\n", NR, $0}}\' '
                    f"'{path}'",
                    timeout=15,
                )
                body = (r.stdout or "").rstrip("\n") if r.success else (
                    f"[view failed: rc={r.returncode} "
                    f"stderr={(r.stderr or '')[:200]!r}]"
                )
                outputs.append(
                    f"Here's the result of running `cat -n` on "
                    f"{path} lines {start}-{end}:\n{body}"
                )
        pending_view_output.clear()
        if outputs:
            pending_view_output.append("\n\n".join(outputs))
        return result

    return submit_result


class LibrarianRoleMixin:
    """Persistent-librarian wiring shared by the librarian runners.

    ``LIBRARIAN_ID`` is the registry / memory-strategy key for the librarian
    sub-agent (``"navigator"`` historically for the standalone librarian,
    ``"librarian"`` for the overlays — kept distinct so existing turn logs
    don't change label).
    """

    LIBRARIAN_ID: str = "librarian"

    #: Leave-one-out ablations of the full librarian (L2). Each removes
    #: exactly one mechanism; ``None`` = full L2 behaviour.
    VALID_LIBRARIAN_ABLATIONS = (None, "persistence", "pointer", "scoped", "freshness")

    def _post_init(self, kwargs: dict[str, Any]) -> None:
        super()._post_init(kwargs)  # MASRunnerBase: drops runtime_image
        self.librarian_ablation = kwargs.pop("librarian_ablation", None)
        if self.librarian_ablation not in self.VALID_LIBRARIAN_ABLATIONS:
            raise ValueError(
                f"librarian_ablation={self.librarian_ablation!r} not in "
                f"{self.VALID_LIBRARIAN_ABLATIONS}"
            )
        self.memory_strategy = kwargs.pop("memory_strategy", "branching")
        self.memory_strategy_kwargs = dict(kwargs.pop("memory_strategy_kwargs", None) or {})
        if not self.drop_below_threshold and self.memory_strategy != "keep_all":
            logger.info(
                "[%s] drop_below_threshold=False overrides memory_strategy=%r → 'keep_all'",
                self.LABEL, self.memory_strategy,
            )
            self.memory_strategy = "keep_all"

    def _make_navigator(self, *, navigator_agent: Any, work_dir: str):
        strategy = build_memory_strategy(
            self.memory_strategy,
            agent_id=self.LIBRARIAN_ID,
            branch_threshold_chars=self.branch_threshold_chars,
            **self.memory_strategy_kwargs,
        )
        return SharedSubAgent(
            agent=navigator_agent,
            docker=self.docker,
            work_dir=work_dir,
            agent_id=self.LIBRARIAN_ID,
            branch_threshold_chars=self.branch_threshold_chars,
            strategy=strategy,
            model=self.model,
            freshness_report=(
                getattr(self, "librarian_ablation", None) != "freshness"
            ),
        )


def make_librarian_call(
    *,
    mr: Any,
    shared_librarian: Any,
    librarian_id: str,
    caller_role: str,
    max_turns: int,
    pending_view_output: list[str],
    max_nudges: int,
    text_only_policy: Any = None,
    swallow_errors: bool = True,
) -> Callable[[str], Awaitable[str]]:
    """Build the persistent-librarian delegation body ``fn(query) -> str``.

    Resumes the one long-lived librarian instance (``UPSERT``) so its history
    accumulates across calls, then prepends any ``view_commands`` output.
    """

    def _record(query: str, reply: str) -> str:
        # Publish this librarian consultation as the caller's last (Action,
        # Observation) pair, keyed on the QUERY (not the librarian's reply,
        # which varies run to run), so the caller's StuckDetectionHooks flag a
        # caller that re-issues the identical query — without collapsing
        # distinct queries (the empty-args on_tool_end fallback would).
        sig = _canonical_args({"query": query})
        _LAST_TOOL_EVENT_PAIR.set((
            ToolCallAction(name=librarian_id, args_json=sig),
            ToolOutputObservation(name=librarian_id, content=sig),
        ))
        return reply

    async def call(query: str) -> str:
        pending_view_output.clear()

        async def _execute(inst, hooks):
            kw: dict[str, Any] = {}
            if text_only_policy is not None:
                kw["text_only_policy"] = text_only_policy
            return await shared_librarian.invoke(
                query, caller_role=caller_role, hooks=hooks,
                max_turns=max_turns, **kw,
            )

        try:
            out = await invoke_agent_logged(
                **mr.logged(
                    agent_type=librarian_id,
                    agent_obj_factory=lambda: shared_librarian,
                    execute=_execute,
                    agent_id=librarian_id,
                    mode=CreateMode.UPSERT,
                    max_nudges=max_nudges,
                )
            )
        except AgentBusyError as e:
            pending_view_output.clear()
            return _record(query, f"cannot consult librarian: {e}")
        except Exception as e:  # noqa: BLE001
            pending_view_output.clear()
            if not swallow_errors:
                raise
            logger.error("librarian failed: %s", e)
            return _record(query, f"librarian failed: {e}")

        views = pending_view_output[0] if pending_view_output else ""
        pending_view_output.clear()
        return _record(query, f"{views}\n\n{out}" if views else out)

    return call


def make_ephemeral_librarian_call(
    *,
    mr: Any,
    executor: Any,
    librarian_agent: Any,
    librarian_id: str,
    max_turns: int,
    pending_view_output: list[str],
    max_nudges: int,
    text_only_policy: Any = None,
    swallow_errors: bool = True,
) -> Callable[[str], Awaitable[str]]:
    """Build the EPHEMERAL librarian delegation body ``fn(query) -> str``.

    "persistence" ablation of :func:`make_librarian_call`: same query-keyed
    stuck pair, error handling, and ``view_commands`` prepend — but each
    call runs the librarian agent fresh (``CreateMode.NEW``, no
    :class:`SharedSubAgent`, no history render, no freshness report).
    """

    def _record(query: str, reply: str) -> str:
        # Same query-keyed (Action, Observation) pair as the persistent
        # path — see make_librarian_call for why the QUERY is the key.
        sig = _canonical_args({"query": query})
        _LAST_TOOL_EVENT_PAIR.set((
            ToolCallAction(name=librarian_id, args_json=sig),
            ToolOutputObservation(name=librarian_id, content=sig),
        ))
        return reply

    async def call(query: str) -> str:
        pending_view_output.clear()

        async def _execute(inst, hooks):
            kw: dict[str, Any] = {"max_turns": max_turns}
            if text_only_policy is not None:
                kw["text_only_policy"] = text_only_policy
            return await executor.execute_one(
                agent=librarian_agent, input=query, hooks=hooks, **kw,
            )

        try:
            result = await invoke_agent_logged(
                **mr.logged(
                    agent_type=librarian_id,
                    agent_obj_factory=lambda: librarian_agent,
                    execute=_execute,
                    mode=CreateMode.NEW,
                    max_nudges=max_nudges,
                )
            )
            # SharedSubAgent.invoke strips reasoning internally; the
            # ephemeral path gets the raw run result and strips here.
            out = strip_subagent_output(result, mr.model)
        except AgentBusyError as e:
            pending_view_output.clear()
            return _record(query, f"cannot consult librarian: {e}")
        except Exception as e:  # noqa: BLE001
            pending_view_output.clear()
            if not swallow_errors:
                raise
            logger.error("librarian failed: %s", e)
            return _record(query, f"librarian failed: {e}")

        views = pending_view_output[0] if pending_view_output else ""
        pending_view_output.clear()
        return _record(query, f"{views}\n\n{out}" if views else out)

    return call
