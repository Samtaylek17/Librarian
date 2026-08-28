"""BOAD next-step tool-output wrapper (shared by boad + librarian families).

Upstream SWE-agent wraps every tool result with ``next_step_template``
before showing it to the model. Idempotent (``_boad_wrapped`` marker).
"""

from __future__ import annotations

_NEXT_STEP_PREFIX = "OBSERVATION:\n"
# (after .strip()). Upstream BOAD's wording was
# no-match) and produces no output; the upstream wording made the
_NEXT_STEP_NO_OUTPUT = "(tool produced no output)"
_NEXT_STEP_FN_CALL_RULES = (
    "\n\n**FUNCTION CALL RULES:** Whenever possible, use the available "
    "subagents to complete the next step. If you do not use a subagent, you "
    "MUST briefly justify why."
)


def _wrap_tools_for_boad_template(
    tools,
    *,
    fn_call_rules: bool,
    fn_call_rules_text: str | None = None,
):
    """Mutate each tool's ``on_invoke_tool`` so its returned string is wrapped
    with upstream BOAD's ``next_step_template`` (and, for the orchestrator,
    a function-call rules suffix).

    ``fn_call_rules_text`` lets callers (e.g. ``librarian``) inject a
    custom redirect-style suffix instead of the BOAD default
    ``_NEXT_STEP_FN_CALL_RULES``. When ``None``, falls back to
    ``_NEXT_STEP_FN_CALL_RULES``. Ignored when ``fn_call_rules`` is False.

    Idempotent: sets a ``_boad_wrapped`` marker so repeated calls don't
    double-wrap.
    """
    suffix = fn_call_rules_text if fn_call_rules_text is not None else _NEXT_STEP_FN_CALL_RULES

    def _make_wrapped(orig):
        async def wrapped(ctx, raw_input):
            out = await orig(ctx, raw_input)
            if not isinstance(out, str):
                return out
            stripped = out.strip()
            body = _NEXT_STEP_NO_OUTPUT if not stripped else out
            wrapped_out = _NEXT_STEP_PREFIX + body
            if fn_call_rules:
                wrapped_out += suffix
            return wrapped_out
        return wrapped

    for tool in tools:
        if getattr(tool, "_boad_wrapped", False):
            continue
        tool.on_invoke_tool = _make_wrapped(tool.on_invoke_tool)
        try:
            setattr(tool, "_boad_wrapped", True)
        except Exception:
            pass
    return tools
