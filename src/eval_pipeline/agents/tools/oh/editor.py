"""OH-style str_replace_editor tool: view / create / str_replace / insert / undo_edit.

Vendored from upstream ``openhands_aci/editor/editor.py::OHEditor``,
ported to Docker shell calls. Output formatting (cat -n line numbering,
truncation notice, view-range warnings) matches upstream byte-for-byte.

Single factory ``make_oh_str_replace_editor``.
"""

from __future__ import annotations

import re

from eval_pipeline.agents.stuck_detection import (
    ErrorObservation,
    ToolCallAction,
    ToolOutputObservation,
)
from eval_pipeline.agents.tools._format import (
    OH_DIR_TRUNCATE_NOTICE,
    OH_MAX_RESPONSE_LEN_CHAR,
    OH_SNIPPET_CONTEXT_WINDOW,
    OH_TEXT_FILE_TRUNCATE_NOTICE,
    _maybe_truncate_oh,
    _oh_make_output,
)
from eval_pipeline.agents.tools._state import (
    _LAST_TOOL_EVENT_PAIR,
    _canonical_args,
)
from eval_pipeline.utils.docker import DockerExecutor


def make_oh_str_replace_editor(
    docker: DockerExecutor, work_dir: str,
    *, max_observation_length: int = OH_MAX_RESPONSE_LEN_CHAR,
):
    from agents import function_tool

    @function_tool
    async def str_replace_editor(
        command: str,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        enable_linting: bool = False,
    ) -> str:
        args_json = _canonical_args(dict(
            command=command, path=path, file_text=file_text,
            view_range=view_range, old_str=old_str, new_str=new_str,
            insert_line=insert_line, enable_linting=enable_linting,
        ))

        # Implementation is a line-for-line port of upstream
        # openhands-aci/editor/editor.py::OHEditor (vendored at
        # to upstream where possible.
        async def _impl() -> str:
            full_path = path if path.startswith("/") else f"{work_dir}/{path}"

            # ── validate_path (upstream OHEditor.validate_path) ─────
            if not full_path.startswith("/"):
                return (
                    f"Error: Invalid `path` parameter: {path}. The path should "
                    f"be an absolute path, starting with `/`."
                )
            is_dir = await docker.dir_exists(full_path)
            exists = is_dir or await docker.file_exists(full_path)
            if command == "create" and exists:
                return (
                    f"Error: Invalid `path` parameter: {path}. File already "
                    f"exists at: {full_path}. Cannot overwrite files using "
                    f"command `create`."
                )
            if command != "create" and not exists:
                return (
                    f"Error: Invalid `path` parameter: {path}. The path "
                    f"{full_path} does not exist. Please provide a valid path."
                )
            if command != "view" and is_dir:
                return (
                    f"Error: Invalid `path` parameter: {path}. The path "
                    f"{full_path} is a directory and only the `view` command "
                    f"can be used on directories."
                )

            if command == "view":
                if is_dir:
                    if view_range is not None:
                        return (
                            "Error: Invalid `view_range` parameter: "
                            f"{view_range}. The `view_range` parameter is not "
                            "allowed when `path` points to a directory."
                        )
                    # Mirror upstream: find -maxdepth 2, exclude hidden at both
                    rh = await docker.exec_shell(
                        f"find -L '{full_path}' -mindepth 1 -maxdepth 1 "
                        f"-name '.*'", timeout=15,
                    )
                    hidden_count = (
                        len([x for x in (rh.stdout or "").strip().split("\n") if x])
                        if (rh.stdout or "").strip() else 0
                    )
                    r = await docker.exec_shell(
                        f"find -L '{full_path}' -maxdepth 2 "
                        f"-not \\( -path '{full_path}/\\.*' -o -path "
                        f"'{full_path}/*/\\.*' \\) | sort",
                        timeout=15,
                    )
                    if not r.success and r.stderr:
                        return f"Error viewing {path}: {r.stderr[:300]}"
                    paths = [p for p in (r.stdout or "").strip().split("\n") if p]
                    formatted = []
                    for p in paths:
                        is_d = await docker.dir_exists(p)
                        formatted.append(f"{p}/" if is_d else p)
                    listing = "\n".join(formatted)
                    header = (
                        f"Here's the files and directories up to 2 levels "
                        f"deep in {full_path}, excluding hidden items:\n"
                    )
                    output = header + listing
                    output = _maybe_truncate_oh(
                        output, max_observation_length, OH_DIR_TRUNCATE_NOTICE,
                    )
                    if hidden_count > 0:
                        output += (
                            f"\n\n{hidden_count} hidden files/directories in "
                            f"this directory are excluded. You can use "
                            f"'ls -la {full_path}' to see them."
                        )
                    return output

                nr = await docker.exec_shell(f"wc -l < '{full_path}'", timeout=10)
                try:
                    num_lines = int((nr.stdout or "0").strip())
                except Exception:
                    num_lines = 0
                # Upstream counts lines via a Python iter; wc -l counts
                tail = await docker.exec_shell(
                    f"tail -c 1 '{full_path}' | od -An -c | tr -d ' \\n'",
                    timeout=5,
                )
                if (tail.stdout or "").strip() and not (tail.stdout or "").strip().endswith("\\n"):
                    num_lines += 1

                if not view_range:
                    r = await docker.exec_shell(
                        f"cat '{full_path}'", timeout=15,
                    )
                    if not r.success:
                        return f"Error viewing {path}: {r.stderr[:300]}"
                    return _oh_make_output(
                        r.stdout or "", str(full_path), 1,
                        max_len=max_observation_length,
                    )

                if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                    return (
                        f"Error: Invalid `view_range` parameter: {view_range}. "
                        f"It should be a list of two integers."
                    )
                start_line, end_line = int(view_range[0]), int(view_range[1])
                if start_line < 1 or (num_lines > 0 and start_line > num_lines):
                    return (
                        f"Error: Invalid `view_range` parameter: {view_range}. "
                        f"Its first element `{start_line}` should be within "
                        f"the range of lines of the file: [1, {num_lines}]."
                    )
                warning = None
                if end_line == -1:
                    end_line = num_lines
                elif num_lines > 0 and end_line > num_lines:
                    warning = (
                        f"We only show up to {num_lines} since there're only "
                        f"{num_lines} lines in this file."
                    )
                    end_line = num_lines
                if end_line < start_line:
                    return (
                        f"Error: Invalid `view_range` parameter: {view_range}. "
                        f"Its second element `{end_line}` should be greater "
                        f"than or equal to the first element `{start_line}`."
                    )

                r = await docker.exec_shell(
                    f"sed -n '{start_line},{end_line}p' '{full_path}'",
                    timeout=15,
                )
                if not r.success:
                    return f"Error viewing {path}: {r.stderr[:300]}"
                # Upstream strips trailing empty line from splitlines rejoin;
                content = "\n".join((r.stdout or "").splitlines())
                output = _oh_make_output(
                    content, str(full_path), start_line,
                    max_len=max_observation_length,
                )
                if warning:
                    output = f"NOTE: {warning}\n{output}"
                return output

            if command == "create":
                if file_text is None:
                    return "Error: parameter `file_text` is required for command `create`"
                wr = await docker.write_file(full_path, file_text)
                if not wr.success:
                    err = (wr.stderr or wr.stdout or "(no error message)").strip()[-300:]
                    return f"Error creating {path}: {err}"
                # Save history for undo_edit — same as upstream's
                await docker.exec_shell(
                    f"cp '{full_path}' '{full_path}.bak'", timeout=5,
                )
                return f"File created successfully at: {full_path}"

            if command == "str_replace":
                if old_str is None:
                    return "Error: parameter `old_str` is required for command `str_replace`"
                if new_str is not None and new_str == old_str:
                    return (
                        "Error: Invalid `new_str` parameter. No replacement "
                        "was performed. `new_str` and `old_str` must be "
                        "different."
                    )
                file_content = await docker.read_file(full_path)
                new_str_resolved = new_str or ""

                def _find_all(content: str, needle: str):
                    pat = re.escape(needle)
                    return [
                        (content.count("\n", 0, m.start()) + 1, m.group(), m.start())
                        for m in re.finditer(pat, content)
                    ]

                occ_old, occ_new = old_str, new_str_resolved
                occurrences = _find_all(file_content, occ_old)
                if not occurrences:
                    # Upstream: strip whitespace on both and try once more.
                    occ_old = old_str.strip()
                    occ_new = new_str_resolved.strip()
                    occurrences = _find_all(file_content, occ_old)
                    if not occurrences:
                        return (
                            f"Error: No replacement was performed, old_str "
                            f"`{old_str}` did not appear verbatim in "
                            f"{full_path}."
                        )
                if len(occurrences) > 1:
                    line_numbers = sorted({ln for ln, _, _ in occurrences})
                    return (
                        f"Error: No replacement was performed. Multiple "
                        f"occurrences of old_str `{old_str}` in lines "
                        f"{line_numbers}. Please ensure it is unique."
                    )

                replacement_line, matched_text, idx = occurrences[0]
                new_file_content = (
                    file_content[:idx] + occ_new
                    + file_content[idx + len(matched_text):]
                )
                await docker.exec_shell(
                    f"cp '{full_path}' '{full_path}.bak'", timeout=5,
                )
                wr = await docker.write_file(full_path, new_file_content)
                if not wr.success:
                    err = (wr.stderr or wr.stdout or "(no error message)").strip()[-300:]
                    return f"Error writing {full_path}: {err}"

                # Snippet echo around the edit site — upstream contract.
                start = max(1, replacement_line - OH_SNIPPET_CONTEXT_WINDOW)
                end = replacement_line + OH_SNIPPET_CONTEXT_WINDOW + occ_new.count("\n")
                sr = await docker.exec_shell(
                    f"sed -n '{start},{end}p' '{full_path}'", timeout=15,
                )
                snippet = sr.stdout or ""
                success = (
                    f"The file {full_path} has been edited. "
                    + _oh_make_output(
                        "\n".join(snippet.splitlines()),
                        f"a snippet of {full_path}",
                        start, max_len=max_observation_length,
                    )
                    + "Review the changes and make sure they are as expected. "
                    "Edit the file again if necessary."
                )
                return success

            if command == "insert":
                if insert_line is None:
                    return "Error: parameter `insert_line` is required for command `insert`"
                if new_str is None:
                    return "Error: parameter `new_str` is required for command `insert`"
                nr = await docker.exec_shell(f"wc -l < '{full_path}'", timeout=10)
                try:
                    num_lines = int((nr.stdout or "0").strip())
                except Exception:
                    num_lines = 0
                if insert_line < 0 or insert_line > num_lines:
                    return (
                        f"Error: Invalid `insert_line` parameter: {insert_line}. "
                        f"It should be within the range of allowed values: "
                        f"[0, {num_lines}]"
                    )

                new_str_lines = new_str.split("\n")
                file_text_old = await docker.read_file(full_path)
                old_lines = (file_text_old or "").split("\n")
                # ends with \n; upstream iterates with enumerate over f so it
                before = old_lines[:insert_line]
                after = old_lines[insert_line:]
                combined_lines = before + new_str_lines + after
                new_file_content = "\n".join(combined_lines)

                await docker.exec_shell(
                    f"cp '{full_path}' '{full_path}.bak'", timeout=5,
                )
                wr = await docker.write_file(full_path, new_file_content)
                if not wr.success:
                    err = (wr.stderr or wr.stdout or "(no error message)").strip()[-300:]
                    return f"Error writing {full_path}: {err}"

                start = max(1, insert_line - OH_SNIPPET_CONTEXT_WINDOW + 1)
                end = min(
                    num_lines + len(new_str_lines),
                    insert_line + OH_SNIPPET_CONTEXT_WINDOW + len(new_str_lines),
                )
                sr = await docker.exec_shell(
                    f"sed -n '{start},{end}p' '{full_path}'", timeout=15,
                )
                snippet = sr.stdout or ""
                return (
                    f"The file {full_path} has been edited. "
                    + _oh_make_output(
                        "\n".join(snippet.splitlines()),
                        "a snippet of the edited file",
                        start, max_len=max_observation_length,
                    )
                    + "Review the changes and make sure they are as expected "
                    "(correct indentation, no duplicate lines, etc). Edit the "
                    "file again if necessary."
                )

            if command == "undo_edit":
                # Our simplified history = single .bak snapshot. Upstream
                r = await docker.exec_shell(
                    f"if [ -f '{full_path}.bak' ]; then "
                    f"  mv '{full_path}.bak' '{full_path}' && "
                    f"  echo 'restored'; "
                    f"else echo 'nobackup'; fi",
                    timeout=5,
                )
                if (r.stdout or "").strip() == "nobackup":
                    return f"Error: No edit history found for {full_path}."
                restored = await docker.read_file(full_path)
                return (
                    f"Last edit to {full_path} undone successfully. "
                    + _oh_make_output(
                        restored or "", str(full_path), 1,
                        max_len=max_observation_length,
                    )
                )

            return (
                f"Error: Unrecognized command {command}. The allowed commands "
                f"are: view, create, str_replace, insert, undo_edit."
            )

        result = await _impl()
        # editor returned a recoverable error (mirrors upstream
        is_err = result.startswith("Error")
        obs_typed = (
            ErrorObservation(content=result) if is_err
            else ToolOutputObservation(name="str_replace_editor", content=result)
        )
        _LAST_TOOL_EVENT_PAIR.set((
            ToolCallAction(name="str_replace_editor", args_json=args_json),
            obs_typed,
        ))
        return result

    return str_replace_editor
