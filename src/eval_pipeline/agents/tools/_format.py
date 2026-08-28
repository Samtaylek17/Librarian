"""OH formatting / truncation constants + helpers.

Vendored from upstream OpenHands ``openhands_aci/editor/{config,prompts}.py``.
Kept separate from the factory bodies so the bash / editor / submit
factories can each import only what they need.
"""

from __future__ import annotations


# OpenHands oh_aci constants — match the vendored upstream exactly.
OH_MAX_RESPONSE_LEN_CHAR = 16000
OH_SNIPPET_CONTEXT_WINDOW = 4

OH_TEXT_FILE_TRUNCATE_NOTICE = (
    "<response clipped><NOTE>Due to the max output limit, only part of this "
    "file has been shown to you. You should retry this tool after you have "
    "searched inside the file with `grep -n` in order to find the line "
    "numbers of what you are looking for.</NOTE>"
)
OH_DIR_TRUNCATE_NOTICE = (
    "<response clipped><NOTE>Due to the max output limit, only part of this "
    "directory has been shown to you. You should use `ls -la` instead to "
    "view large directories incrementally.</NOTE>"
)


def _truncate(text: str, max_len: int = 30000) -> str:
    return text


def _maybe_truncate_oh(
    content: str,
    max_len: int = OH_MAX_RESPONSE_LEN_CHAR,
    truncate_notice: str = OH_TEXT_FILE_TRUNCATE_NOTICE,
) -> str:
    """Upstream-matching truncation: cut at ``max_len`` chars and append notice."""
    if not max_len or len(content) <= max_len:
        return content
    return content[:max_len] + truncate_notice


def _oh_make_output(
    snippet_content: str,
    snippet_description: str,
    start_line: int = 1,
    max_len: int = OH_MAX_RESPONSE_LEN_CHAR,
) -> str:
    """Render a snippet the way upstream ``OHEditor._make_output`` does:
    ``cat -n`` style line numbering, prefixed by a short description.

    Matches the upstream implementation byte-for-byte (6-wide line
    number + tab + content) so the model sees the same shape here as
    in native OpenHands.
    """
    snippet_content = _maybe_truncate_oh(snippet_content, max_len)
    lines = snippet_content.split("\n")
    numbered = "\n".join(
        f"{i + start_line:6}\t{line}" for i, line in enumerate(lines)
    )
    return (
        f"Here's the result of running `cat -n` on {snippet_description}:\n"
        f"{numbered}\n"
    )


def _truncate_bash_output(
    output: str, max_len: int = OH_MAX_RESPONSE_LEN_CHAR,
) -> str:
    if not max_len or len(output) <= max_len:
        return output
    tail = output[-max_len:]
    n_lines = tail.count("\n") + 1
    return (
        f"[Previous command outputs are truncated. Showing the last "
        f"{n_lines} lines (~{max_len} chars) of the output below.]\n{tail}"
    )
