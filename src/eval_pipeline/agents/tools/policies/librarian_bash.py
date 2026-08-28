from __future__ import annotations

import os
import re

from eval_pipeline.agents.stuck_detection import (
    CmdOutputObservation,
    CmdRunAction,
)
from eval_pipeline.agents.tools._format import (
    OH_MAX_RESPONSE_LEN_CHAR,
    _truncate_bash_output,
)
from eval_pipeline.agents.tools._state import _LAST_TOOL_EVENT_PAIR
from eval_pipeline.utils.docker import DockerExecutor


_SEARCH_VERBS = frozenset({
    "grep", "rg", "egrep", "fgrep",
    "find",
    "cat", "head", "tail",
    "sed", "awk",
    "less", "more",
    "ls", "tree",
})

_SOURCE_EXTS = (
    ".py", ".pyx", ".pyi", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".js", ".jsx", ".ts", ".tsx",
    ".rs", ".go", ".java", ".kt",
    ".rb", ".sh", ".lua",
    ".html", ".css", ".scss", ".vue",
    ".yml", ".yaml", ".toml", ".cfg", ".ini",
    ".md", ".rst", ".txt",
)


def _looks_like_source_path(tok: str, work_dir: str) -> bool:
    if tok.startswith("-"):
        return False
    raw = tok.strip("'\"")
    if not raw:
        return False
    abs_p = raw if raw.startswith("/") else f"{work_dir}/{raw}"
    norm = os.path.normpath(abs_p)
    wd_abs = os.path.normpath(work_dir).rstrip("/") + "/"
    if not (norm.startswith(wd_abs) or norm == os.path.normpath(work_dir)):
        return False
    if norm.endswith(_SOURCE_EXTS):
        return True
    base = os.path.basename(norm)
    return "." not in base


def _detect_search_verb(command: str, work_dir: str) -> str | None:
    for sub in re.split(r";|\|\||&&|\|", command):
        sub = sub.strip()
        if not sub:
            continue
        parts = sub.split()
        if not parts:
            continue
        first = parts[0]
        if first == "git" and len(parts) >= 2 and parts[1] == "grep":
            return "git grep"
        if first not in _SEARCH_VERBS:
            continue
        if first in {"grep", "rg", "egrep", "fgrep", "find"}:
            return first
        for tok in parts[1:]:
            if _looks_like_source_path(tok, work_dir):
                return first
        if first in {"ls", "tree"} and not [
            t for t in parts[1:] if not t.startswith("-")
        ]:
            return first
    return None


def _warn_message(verb: str) -> str:
    return (
        f"⚠ NOTICE: `{verb}` is a search/read on the repository. The "
        f"command was executed THIS TIME, but for repeated lookups "
        f"prefer ``repo_librarian(query)`` — it accumulates memory "
        f"across calls (so follow-ups are cheap), translates "
        f"natural-language questions into the right grep/view, and "
        f"returns code excerpts at file:line. Phrase your query as a "
        f"lookup question (\"Where is X?\", \"Show me the body of Y\"); "
        f"the librarian only returns code, not analysis.\n"
        f"--- command output below ---\n"
    )


def make_librarian_bash_tool(
    docker: DockerExecutor, work_dir: str,
    *,
    max_observation_length: int = OH_MAX_RESPONSE_LEN_CHAR,
    timeout: int = 1800,
):
    from agents import function_tool

    @function_tool
    async def bash(command: str) -> str:
        verb = _detect_search_verb(command, work_dir)
        warn_prefix = _warn_message(verb) if verb else ""
        r = await docker.exec_in_persistent_shell(command, timeout=timeout)
        output = ""
        if r.stdout:
            output += r.stdout
        if r.stderr:
            output += ("\n" if output else "") + r.stderr
        if not output.strip():
            output = f"(command exited with code {r.returncode})"
        _LAST_TOOL_EVENT_PAIR.set((
            CmdRunAction(command=command),
            CmdOutputObservation(
                command=command, exit_code=int(r.returncode), content=output,
            ),
        ))
        truncated = _truncate_bash_output(output, max_len=max_observation_length)
        return warn_prefix + truncated if warn_prefix else truncated

    return bash
