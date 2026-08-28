from __future__ import annotations

import hashlib
import os
import re as _re

from eval_pipeline.agents.history.view_state import ViewedLinesTracker
from eval_pipeline.utils.docker import DockerExecutor


_DIFF_HUNK_RX = _re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
# Split a unified diff into one chunk per file (each starts with "diff --git").
_FILE_SPLIT_RX = _re.compile(r"(?m)^(?=diff --git )")


def _parse_diff(diff_text: str) -> dict[str, dict]:
    """Parse ``git diff`` into ``{path: {"ranges": [...], "fp": <hash>}}``.

    ``ranges`` are the modified line ranges on the new (``+``) side, taken
    verbatim from the hunk headers — line locations are always derived from
    ``git diff``. ``fp`` is a hash of the file's entire diff section, which
    changes on *any* write to the file's content — including an in-place
    re-edit that preserves the line range (the ``index <sha>..<sha>`` line and
    the ``+`` body both change). It is the signal used to decide whether a
    file has been written since it was last shown.
    """
    out: dict[str, dict] = {}
    for chunk in _FILE_SPLIT_RX.split(diff_text):
        if not chunk.strip():
            continue
        path: str | None = None
        ranges: list[tuple[int, int]] = []
        for ln in chunk.splitlines():
            if ln.startswith("+++ b/"):
                path = ln[len("+++ b/"):].strip()
            elif ln.startswith("@@"):
                m = _DIFF_HUNK_RX.match(ln)
                if m:
                    start = int(m.group(1))
                    count = int(m.group(2)) if m.group(2) else 1
                    ranges.append(
                        (start, start) if count == 0
                        else (start, start + count - 1)
                    )
        if not path or path == "/dev/null":
            continue
        fp = hashlib.sha1(chunk.encode("utf-8", "replace")).hexdigest()
        out[path] = {"ranges": ranges, "fp": fp}
    return out


async def build_freshness_report(
    docker: DockerExecutor,
    work_dir: str,
    last_shown_diff: dict[str, str],
    viewed_lines: ViewedLinesTracker,
) -> str:
    try:
        diff_r = await docker.exec_shell(
            f"cd {work_dir} && git diff -U0 HEAD",
        )
        diff_text = diff_r.stdout or ""
    except Exception:
        diff_text = ""

    cur = _parse_diff(diff_text)

    def _reset(path: str) -> None:
        viewed_lines.reset_file(path)
        viewed_lines.reset_file(f"/testbed/{path}")
        viewed_lines.reset_file(os.path.join(work_dir, path))

    # A file is shown when its content fingerprint is new or differs from what
    # was last reported — i.e. it has been WRITTEN since it was last shown.
    # Because the fingerprint hashes the whole diff section, a same-range
    # in-place re-edit (which the line ranges alone could not distinguish)
    # changes it and re-fires the report.
    new_edits: dict[str, list[tuple[int, int]]] = {}
    for path, info in cur.items():
        if last_shown_diff.get(path) != info["fp"]:
            new_edits[path] = info["ranges"]
            last_shown_diff[path] = info["fp"]
            _reset(path)

    # Files that were reported earlier but are no longer in the diff have been
    # reverted to the original commit — also a write since the last report, so
    # invalidate any cached excerpt of them.
    reverted = [p for p in list(last_shown_diff) if p not in cur]
    for path in reverted:
        last_shown_diff.pop(path, None)
        _reset(path)

    if not new_edits and not reverted:
        return (
            "[FILE FRESHNESS REPORT — no changes since the previous wake. "
            "Quote from your history instead of re-viewing any file.]"
        )

    lines: list[str] = []
    for path in sorted(new_edits):
        ranges = new_edits[path]
        if not ranges:
            lines.append(f"  - {path}: (file present in diff but no hunks)")
        else:
            rs = ", ".join(
                f"{a}-{b}" if a != b else str(a) for a, b in ranges
            )
            lines.append(f"  - {path}: lines [{rs}]")
    for path in sorted(reverted):
        lines.append(
            f"  - {path}: reverted to the original commit — "
            "discard any cached excerpt of it"
        )
    return (
        "[FILE FRESHNESS REPORT — files WRITTEN since the previous wake. "
        "For each listed file, content OUTSIDE the given ranges is unchanged "
        "from when you last viewed it — quote from history. Re-view a listed "
        "range only if your prior view intersects it. Files NOT listed are "
        "unchanged since the previous report.]\n"
        + "\n".join(lines)
    )
