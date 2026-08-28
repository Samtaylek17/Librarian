"""Tool-call snippet formatting per model profile.

System prompts can embed tool-call examples in canonical Python form::

    bash(command="grep -rn ...")
    str_replace_editor(command="view", path="...", view_range=[1, 50])
    delegate_to_navigator(request="your detailed request")

Different model families expect different on-wire formats:

* OpenAI / Qwen3 — Python form matches their tool-call rendering, leave as-is.
* Seed-OSS — convert to the chat-template's XML form::

      <seed:tool_call>
      <function=bash>
      <parameter=command>grep -rn ...</parameter>
      </function>
      </seed:tool_call>

Each :class:`ModelProfile` carries a :data:`tool_call_renderer` that the
MAS modules apply to their per-role system prompts before constructing
the ``Agent``. This module provides:

- :data:`KNOWN_TOOLS` — the closed set of tool names whose snippets get
  rewritten. Anything else (markdown ```code blocks of unrelated python``
  or prose containing ``some_func(...)``) is left untouched.
- :func:`identity_renderer` — no-op default for OpenAI-compatible models.
- :func:`seed_oss_renderer` — Python → XML rewrite for Seed-OSS.

Snippets are located via a regex anchored on the known tool names, then
parsed with :mod:`ast`. We use :func:`ast.get_source_segment` to copy
each value's verbatim source slice (rather than ``ast.literal_eval`` —
that would corrupt embedded escape sequences such as ``\\b`` in a grep
pattern). Snippets that do not parse as a simple call to a Name with
only kwargs (e.g. prose like ``submit_result(result=…)`` with a unicode
ellipsis) are skipped — best-effort rewrite, never destructive.
"""
from __future__ import annotations

import ast
import re
from typing import Callable

KNOWN_TOOLS: tuple[str, ...] = (
    "bash",
    "str_replace_editor",
    "delegate_to_navigator",
    "delegate_to_editor",
    "delegate_to_executor",
    "submit_result",
    "finish",
)


_PATTERN = re.compile(r"\b(" + "|".join(KNOWN_TOOLS) + r")\(")


def _find_balanced_close(text: str, open_idx: int) -> int:
    depth = 1
    i = open_idx + 1
    in_str: str | None = None
    while i < len(text):
        c = text[i]
        if in_str is not None:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
        elif c in ('"', "'"):
            in_str = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _parse_call(snippet: str) -> tuple[str, list[tuple[str, str, bool]]]:
    """Parse ``name(k=v, ...)`` into ``(name, [(key, source_seg, is_string)])``.

    ``source_seg`` is the verbatim source slice of the value (escape
    sequences NOT processed). For string literals, ``is_string`` is
    True and the segment will include the outer quotes — strip them
    when emitting XML, keep them when re-emitting Python.

    Raises:
        SyntaxError, ValueError, AttributeError if the snippet is not a
        simple call to a Name with only keyword arguments (e.g. it has
        positional args, ``**kwargs``, or is not a Call at all).
    """
    expr = ast.parse(snippet, mode="eval")
    call = expr.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError("not a simple function call")
    if call.args:
        raise ValueError("positional arguments not supported")
    name = call.func.id
    kwargs: list[tuple[str, str, bool]] = []
    for kw in call.keywords:
        if kw.arg is None:
            raise ValueError("**kwargs not supported")
        seg = ast.get_source_segment(snippet, kw.value)
        if seg is None:
            raise ValueError(f"could not retrieve source segment for {kw.arg}")
        is_string = (
            isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)
        )
        kwargs.append((kw.arg, seg, is_string))
    return name, kwargs


def _render_seed_oss(name: str, kwargs: list[tuple[str, str, bool]]) -> str:
    out = ["<seed:tool_call>", f"<function={name}>"]
    for k, seg, is_str in kwargs:
        if is_str and len(seg) >= 2 and seg[0] in ('"', "'") and seg[-1] == seg[0]:
            value = seg[1:-1]
        else:
            value = seg
        out.append(f"<parameter={k}>{value}</parameter>")
    out.append("</function>")
    out.append("</seed:tool_call>")
    return "\n".join(out)


def identity_renderer(text: str) -> str:
    return text


def seed_oss_renderer(text: str) -> str:
    pieces: list[str] = []
    cursor = 0
    for m in _PATTERN.finditer(text):
        if m.start() < cursor:
            continue
        open_idx = m.end() - 1
        close_idx = _find_balanced_close(text, open_idx)
        if close_idx < 0:
            continue
        snippet = text[m.start():close_idx]
        try:
            name, kwargs = _parse_call(snippet)
        except (SyntaxError, ValueError, AttributeError):
            continue
        pieces.append(text[cursor:m.start()])
        pieces.append(_render_seed_oss(name, kwargs))
        cursor = close_idx
    pieces.append(text[cursor:])
    return "".join(pieces)


ToolCallRenderer = Callable[[str], str]


__all__ = [
    "KNOWN_TOOLS",
    "ToolCallRenderer",
    "identity_renderer",
    "seed_oss_renderer",
]
