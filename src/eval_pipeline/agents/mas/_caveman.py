"""Caveman output-style overlay — appends a terse-output directive to a
sub-agent's system prompt.

Source: github.com/juliusbrussee/caveman (skills/caveman/SKILL.md). Idea is
a stylistic compression of model output ("drop filler, keep substance") —
targets ``out_tok`` without changing reasoning or tool semantics.

Used by ``mas/boad.py`` and ``mas/hyperagent.py`` when ``caveman=True`` is
passed at runner construction. The directive is APPENDED to each
sub-agent's existing role-specific system prompt so tool-use contracts and
role boundaries stay intact; only the natural-language style is altered.

Levels mirror the upstream skill — ``full`` is the verified default.
"""

from __future__ import annotations


# Verbatim "Rules" + tightened "Auto-Clarity" carve-outs from upstream
_CAVEMAN_FULL = """\
OUTPUT STYLE — caveman mode (FULL).

Respond terse like smart caveman. All technical substance stays. Only \
fluff dies. ACTIVE every response until task done.

Rules:
- Drop articles (a/an/the), filler (just/really/basically/actually/simply), \
pleasantries (sure/certainly/of course/happy to), hedging.
- Fragments OK. Short synonyms (big not extensive, fix not "implement a \
solution for").
- Pattern: `[thing] [action] [reason]. [next step].`

NEVER compress / NEVER abbreviate:
- Code blocks, code symbols, function names, class names, file paths, \
error strings, log lines, shell commands, tool-call arguments.
- Anything inside `backticks`, fenced code, or quoted strings.
- Patch / diff content (must be byte-exact).

Resume normal prose for: security warnings, irreversible-action \
confirmations, multi-step sequences where fragment order could be \
misread.

Example
- Normal: "The reason your React component is re-rendering is likely \
because you're creating a new object reference on each render cycle. \
You should wrap it in useMemo."
- Caveman: "New object ref each render. Inline object prop = new ref = \
re-render. Wrap in `useMemo`."\
"""


_CAVEMAN_LITE = """\
OUTPUT STYLE — caveman LITE.

Drop filler (just/really/basically/actually/simply), pleasantries (sure/\
certainly/happy to), hedging. Keep articles and full sentences. \
Professional but tight.

NEVER abbreviate code symbols, function names, file paths, error \
strings, or tool-call arguments. Patch / diff content must be byte-exact.\
"""


_CAVEMAN_ULTRA = """\
OUTPUT STYLE — caveman ULTRA.

Maximum prose compression. Drop articles + filler + hedging + \
conjunctions. Abbreviate common prose words (DB/auth/config/req/res/fn/\
impl). Use arrows for causality (X → Y). One word when one word enough.

NEVER abbreviate / NEVER compress: code symbols, function names, class \
names, file paths, error strings, log lines, shell commands, tool-call \
arguments, patch / diff content. Anything inside backticks or fenced \
code stays byte-exact.

Resume normal prose for security warnings + irreversible-action \
confirmations.\
"""


_LEVELS = {
    "lite": _CAVEMAN_LITE,
    "full": _CAVEMAN_FULL,
    "ultra": _CAVEMAN_ULTRA,
}


def apply_caveman(prompt: str, level: str = "full") -> str:
    """Append the caveman directive for ``level`` to an existing system prompt.

    Append (not replace): the role's existing rules — tool-use contract,
    stop-tool, "stay in your lane" — stay authoritative. Caveman only
    rewrites the prose style.

    ``level`` is one of ``lite``/``full``/``ultra``; default ``full``
    matches upstream SKILL.md's own default.
    """
    directive = _LEVELS.get(level)
    if directive is None:
        raise ValueError(
            f"Unknown caveman level {level!r}; expected one of "
            f"{sorted(_LEVELS)}"
        )
    return f"{prompt}\n\n{directive}"
