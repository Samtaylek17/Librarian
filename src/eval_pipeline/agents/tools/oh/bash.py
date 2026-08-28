from __future__ import annotations

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


def make_oh_bash(
    docker: DockerExecutor, work_dir: str,
    *,
    max_observation_length: int = OH_MAX_RESPONSE_LEN_CHAR,
    timeout: int = 1800,
):
    from agents import function_tool

    @function_tool
    async def bash(command: str) -> str:
        r = await docker.exec_in_persistent_shell(command, timeout=timeout)
        output = ""
        if r.stdout:
            output += r.stdout
        if r.stderr:
            output += ("\n" if output else "") + r.stderr
        if not output.strip():
            output = f"(no output, exit code {r.returncode})"
        elif r.returncode not in (0, -1):
            # Non-zero exit with non-empty output: append the upstream
            output = f"{output}\n[Command finished with exit code {r.returncode}]"
        # upstream behavior — the model still sees the truncated string.
        _LAST_TOOL_EVENT_PAIR.set((
            CmdRunAction(command=command),
            CmdOutputObservation(
                command=command, exit_code=int(r.returncode), content=output,
            ),
        ))
        return _truncate_bash_output(output, max_len=max_observation_length)

    return bash
