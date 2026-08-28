from __future__ import annotations

import os
import re


_SCOPED_SOURCE_EXTS = (
    ".py", ".pyx", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".rb", ".php", ".cs", ".swift",
    ".kt", ".scala", ".sh", ".lua", ".sql",
    ".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".toml", ".cfg",
    ".ini", ".xml", ".html", ".css",
)
_BASH_FILE_READ_VERBS = {"cat", "head", "tail", "less", "more", "sed", "awk"}


_AUTO_ROUTE_QUERY_TMPL = (
    "Please answer from your prior session history if you already have "
    "this file's content cached — do NOT re-execute the bash command "
    "verbatim against the file. Caller's original intent was captured "
    "as the following bash command:\n\n"
    "    {command}\n\n"
    "Infer the target region / aspect the caller was after (e.g. a "
    "specific line range, a method body, a search for a symbol) and "
    "answer concisely based on what you already know about the file. "
    "If the answer is genuinely not in your cached history, then — and "
    "only then — read freshly."
)


def _build_auto_route_query(command: str) -> str:
    return _AUTO_ROUTE_QUERY_TMPL.format(command=command.strip())


def _parse_command_range(command: str) -> tuple[int, int] | None:
    s = command.strip()
    m = re.search(r"sed\s+-n\s+['\"]?(\d+),(\d+)\s*p['\"]?", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m = re.search(r"head\s+-n?\s*(\d+)", s)
    if m:
        return (1, int(m.group(1)))
    return None


def _find_bash_target_file(command: str, work_dir: str) -> str | None:
    for sub in re.split(r';|\|\||&&|\|', command):
        sub = sub.strip()
        if not sub:
            continue
        parts = sub.split()
        if not parts:
            continue
        first = parts[0]
        if first not in _BASH_FILE_READ_VERBS:
            continue
        for tok in parts[1:]:
            tok = tok.strip("'\"")
            if tok.startswith("-"):
                continue
            if not tok.endswith(_SCOPED_SOURCE_EXTS):
                continue
            return tok
    return None


def _check_orchestrator_bash(command: str, work_dir: str) -> str | None:
    wd_abs = os.path.normpath(work_dir).rstrip("/") + "/"

    def _norm(tok: str) -> str:
        ap = tok if tok.startswith("/") else f"{work_dir}/{tok}"
        return os.path.normpath(ap)

    for sub in re.split(r';|\|\||&&|\|', command):
        sub = sub.strip()
        if not sub:
            continue
        parts = sub.split()
        if not parts:
            continue
        first = parts[0]
        if first in _BASH_FILE_READ_VERBS:
            for tok in parts[1:]:
                tok = tok.strip("'\"")
                if tok.startswith("-"):
                    continue
                if not tok.endswith(_SCOPED_SOURCE_EXTS):
                    continue
                norm = _norm(tok)
                if norm.startswith(wd_abs) or norm == os.path.normpath(work_dir):
                    return (
                        f"REJECTED: cannot bash `{first} {tok}` — agents "
                        f"do NOT read source files directly. Pass your "
                        f"plan / question / edit purpose to "
                        f"ask_shared_layer('{tok}', '<intent>'). The "
                        f"shared layer (a per-file analyst that already "
                        f"knows this file) will execute and report back."
                    )
    _exts_alt = "|".join(ext.lstrip(".") for ext in _SCOPED_SOURCE_EXTS)
    py_open_rx = re.compile(
        r"\bpython3?\b[\s\S]*?open\s*\(\s*['\"]([^'\"]+\.(?:" + _exts_alt + r"))['\"]",
        re.S,
    )
    m = py_open_rx.search(command)
    if m:
        path = m.group(1)
        norm = _norm(path)
        if norm.startswith(wd_abs):
            return (
                f"REJECTED: `python -c` (or python script) opening "
                f"'{path}' is treated as a single-source-file read — "
                f"the same restriction as cat/sed/head/etc. Pass your "
                f"plan / question to ask_shared_layer('{path}', "
                f"'<intent>'). The shared layer reads/edits the file "
                f"on your behalf and reports back."
            )
    return None


_LINE_NUM_PREFIX_RX = re.compile(r"^\s*\d+[\t:]\s?")


def _looks_like_file_content(output: str) -> bool:
    body = output.strip()
    if not body:
        return False
    lines = body.split("\n")
    nonempty = [l for l in lines if l.strip()]
    if not nonempty:
        return False
    prefixed = sum(1 for l in nonempty if _LINE_NUM_PREFIX_RX.match(l))
    if len(nonempty) >= 5 and prefixed / len(nonempty) >= 0.3:
        return True
    if len(nonempty) >= 10:
        sample = "\n".join(nonempty[:80]).lower()
        bad_markers = ("traceback", "failed", "error:", "warning:",
                       "passed", "skipped", "==", "fail", "not ok",
                       "{", "[")
        if any(m in sample[:200] for m in ("traceback", "==")) :
            return False
        if sample.startswith(("{", "[")):
            return False
        return True
    return False
