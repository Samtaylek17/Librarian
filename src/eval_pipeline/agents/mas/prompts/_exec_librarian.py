"""Exec-Librarian prompts — ONLY the Librarian's persistent-session
mechanism applied to the HyperAgent Executor role.

Everything else stays the verbatim upstream Executor: same ``bash`` +
``submit_result(result)`` contract (no pointer-only ``run_commands``
channel), same scope wording, no freshness-report reinterpretation.
The single change is persistence: one Executor instance per episode
whose history accumulates across delegations, plus the prompt text that
tells the Executor (and the planner) how to exploit that memory.

Built by assert-guarded surgery on the verbatim upstream prompts
(:data:`mas.prompts._hyperagent._EXECUTOR_SYSTEM`) so drift in the
source fails at import time, matching the discipline of
:mod:`._librarian`.
"""
from __future__ import annotations

from eval_pipeline.agents.mas.prompts._hyperagent import (
    _EXECUTOR_SYSTEM,
    _PLANNER_SYSTEM as _PLANNER_SYSTEM_HA_BASE,
)


# ── Executor system prompt (original + persistence semantics) ────────────

_PERSISTENCE_BLOCK = """\

### Persistence:
You are the SAME Executor instance across every delegation in this
episode. Your full conversation history is visible when you wake up —
prior commands, their outputs, repro scripts you wrote, environment
quirks you worked around, all of it. BEFORE re-deriving how to run
something (test invocation syntax, conda activation, working
directory), scan your history: if you already ran it, reuse the exact
command. Scripts you wrote earlier (e.g. ``/tmp/repro.py``) still exist
on disk — re-run them, do not rewrite them. When the planner indicates
the code has changed since your previous run (e.g. "the fix is applied"),
re-run the relevant command instead of quoting its old outcome."""

_EXEC_LIBRARIAN_SYSTEM = _EXECUTOR_SYSTEM + _PERSISTENCE_BLOCK

# ── Planner-facing executor policy (persistence only) ────────────────────

EXECUTOR_POLICY = """\
Executor policy (read this BEFORE you delegate a run):
  - The Executor is a single long-lived instance: its session persists across every ``delegate_to_executor`` call in this episode. It remembers how this repo's tests are invoked, the repro scripts it already wrote, the environment quirks it already worked around, and every prior run's outcome — so follow-up runs are cheap. Do NOT re-explain the environment or repeat instructions it has already received; state WHAT to run or verify, and include file/test context only when it is NEW.
  - Each NEW verification need (a test to run, a repro to execute, runtime behaviour to observe) is a NEW ``delegate_to_executor`` call. Re-running a previously failing test after an edit is exactly such a call — the Executor already knows the command; say which result you need refreshed (e.g. "the fix is applied; re-run the test that failed before")."""


def _insert_before_guidelines(planner_system: str) -> str:
    marker = "### Guidelines:"
    assert planner_system.count(marker) == 1, "planner Guidelines marker drifted"
    return planner_system.replace(
        marker, "### " + EXECUTOR_POLICY + "\n\n" + marker,
    )


#: HA baseline planner + executor policy (navigator / editor untouched).
_PLANNER_SYSTEM_EXEC = _insert_before_guidelines(_PLANNER_SYSTEM_HA_BASE)

# LLM-facing description for the persistent `delegate_to_executor` tool
DELEGATE_TO_EXECUTOR_DOC = """Delegate a command/test execution to the Executor.

The Executor runs shell commands — test suites, repro scripts,
runtime inspections — and reports the outcome. It IS persistent:
its session accumulates across every call within this episode,
so it remembers this repo's test syntax, the repro scripts it
wrote, and every prior result — follow-up runs are cheaper than
first runs.

Args:
    request: What to run or verify. Good shapes: "Write and
        run a minimal repro for <issue behaviour>.", "Run the
        test file at <path>.", "The fix is applied — re-run
        the test that failed before." Include file/test
        context only when it is NEW — the Executor remembers
        prior context.
"""
