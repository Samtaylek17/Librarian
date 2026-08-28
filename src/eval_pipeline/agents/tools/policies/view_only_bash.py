from __future__ import annotations

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


_BASH_EDIT_VERB_RE = re.compile(
    r"\bsed\b[^|;]*?-i\b"
    r"|(?<![>&])>(?!\s*&)(?!\s*/dev/null)\s*[\w\-/\.]"
    r"|(?<![>&])>>(?!\s*&)(?!\s*/dev/null)\s*[\w\-/\.]"
    r"|^\s*(?:cp|mv|rm|touch|chmod|chown|patch|tee)\b"
)


def make_view_only_bash(
    docker: DockerExecutor, work_dir: str,
    *,
    max_observation_length: int = OH_MAX_RESPONSE_LEN_CHAR,
    timeout: int = 1800,
):
    from agents import function_tool

    @function_tool
    async def bash(command: str) -> str:
        if _BASH_EDIT_VERB_RE.search(command):
            return (
                f"REJECTED: this bash command appears to mutate files. "
                f"Navigator may only READ. If an edit is needed, "
                f"finish via submit_result and let the Planner "
                f"delegate to the Editor.\nCommand: {command[:200]}"
            )
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
        return _truncate_bash_output(output, max_len=max_observation_length)

    return bash
