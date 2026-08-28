"""OpenHands CodeActAgent system prompt loader.

Loads the vendored ``system_prompt.j2`` (third_party/openhands_vendored/
prompts/, pinned via ``COMMIT_PIN.txt``) and returns it as plain text.
The upstream is a Jinja2 template; everything we need is plain text
except a single ``{% include 'security_risk_assessment.j2' %}`` directive
that we strip (we don't run OH's security policy system).

This is **one** prompt source among many. BOAD / HyperAgent / SharedFS
each have their own prompts in ``mas/<topology>.py`` or in this
``prompts/`` package. Nothing forces a runner to pick OH — it just
happens to be the default for the OH-mimic agents (vanilla, sharedfs,
hyperagent sub-agents, BOAD orchestrator).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_OH_SYSTEM_PROMPT_FILE = (
    _REPO_ROOT / "third_party/openhands_vendored/prompts/system_prompt.j2"
)


def load_oh_system_prompt() -> str:
    """Return the verbatim vendored OpenHands CodeActAgent system prompt."""
    try:
        text = _OH_SYSTEM_PROMPT_FILE.read_text()
    except FileNotFoundError:
        logger.warning(
            "OH system_prompt.j2 not found at %s — falling back to a stub",
            _OH_SYSTEM_PROMPT_FILE,
        )
        return _FALLBACK_SYSTEM_PROMPT

    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("{%") and "security_risk_assessment" in stripped:
            continue
        lines.append(line)
    return "\n".join(lines)


_load_oh_system_prompt = load_oh_system_prompt


_FALLBACK_SYSTEM_PROMPT = """\
You are a software engineer fixing issues in code repositories at {work_dir}.

<EFFICIENCY>
Each action is expensive. Combine multiple bash commands into one when possible.
Use find/grep to locate files efficiently before reading them.
</EFFICIENCY>

<WORKFLOW>
1. EXPLORE: Understand repo structure with find, grep, git log
2. ANALYZE: Read relevant files with str_replace_editor view
3. IMPLEMENT: Make minimal, focused changes with str_replace_editor str_replace
4. TEST: Run tests to verify your fix
5. COMMIT: git add -A && git commit -m "description"
6. Call submit to finish
</WORKFLOW>
"""
