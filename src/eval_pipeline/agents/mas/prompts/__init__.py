"""Verbatim sub-agent prompts for the MAS runners (reference layer).

Prompt bodies live here so each runner file stays a high-level wiring
view (Config + SUBAGENTS table + wire()). Byte-identical to the paper /
upstream sources — do NOT reflow; runs are compared against these exact
strings.

Submodules:
    _boad        — BOAD orchestrator + issue_analyzer + code_navigator,
                   plan-gen / fixed plan, next-step docstrings.
    _hyperagent  — HyperAgent planner + Navigator/Editor/Executor.
    _librarian   — repo-Librarian system prompt, tool-selection policies,
                   OH main-agent instance template, redirect suffix.
    _variants    — librarian-overlay planner / fixed-plan prompts that
                   compose the three families above.
"""
