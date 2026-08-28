from __future__ import annotations

import json as _json

from eval_pipeline.agents.tools._format import OH_MAX_RESPONSE_LEN_CHAR
from eval_pipeline.agents.tools._oh_body import make_tools
from eval_pipeline.utils.docker import DockerExecutor


def make_view_only_str_replace_editor(
    docker: DockerExecutor, work_dir: str,
    *, max_observation_length: int = OH_MAX_RESPONSE_LEN_CHAR,
):
    from agents import function_tool

    base_tools = {
        getattr(t, "name", None): t
        for t in make_tools(
            docker, work_dir, max_observation_length=max_observation_length,
        )
    }
    full_sre = base_tools["str_replace_editor"]

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
        if command != "view":
            return (
                f"REJECTED: str_replace_editor command={command!r} is "
                f"not allowed for the Navigator. Navigator may only "
                f"VIEW files. To EDIT, finish your finding with "
                f"submit_result and let the Planner delegate to the "
                f"Editor."
            )
        from agents.tool_context import ToolContext
        args_json = _json.dumps({
            "command": command, "path": path, "file_text": file_text,
            "view_range": view_range, "old_str": old_str, "new_str": new_str,
            "insert_line": insert_line, "enable_linting": enable_linting,
        })
        ctx = ToolContext(
            context=None,
            tool_name="str_replace_editor",
            tool_call_id="view-only",
            tool_arguments=args_json,
        )
        return await full_sre.on_invoke_tool(ctx, args_json)

    return str_replace_editor
