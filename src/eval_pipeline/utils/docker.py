from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass

# layout follows the OpenHands V0 ``BashSession`` pattern, vendored at
# ``third_party/openhands_legacy_vendored/bash.py`` (COMMIT_PIN'd).
from openhands.tools.terminal.constants import CMD_OUTPUT_PS1_END
from openhands.tools.terminal.metadata import CmdOutputMetadata

from eval_pipeline.utils._bash_helpers import (
    escape_bash_special_chars,
    split_bash_commands,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    success: bool
    returncode: int
    stdout: str
    stderr: str


class DockerExecutor:
    """Execute commands in a Docker container.

    Provides two execution modes:

    - ``exec`` / ``exec_shell``: stateless one-shot ``docker exec`` per
      call. env vars, cwd, virtualenv activation are NOT preserved
      between calls.
    - ``exec_in_persistent_shell``: stateful interactive bash session
      inside the container, kept alive in a tmux pane. cwd, env,
      activated venv all persist across calls.

    The persistent-shell layer follows OpenHands V0's ``BashSession``
    pattern (``third_party/openhands_legacy_vendored/bash.py``):

      * One tmux session per ``DockerExecutor`` instance, lazily created.
      * Bash inside the pane emits the OH-style sentinel
        ``\\n###PS1JSON###{...$?...}###PS1END###`` at every prompt
        (carries ``exit_code`` and ``working_dir``).
      * Each call: snapshot the initial PS1 count via ``tmux capture-pane``,
        send the (single-statement) command via ``tmux send-keys -l`` +
        ``Enter``, then poll ``capture-pane`` every ``POLL_INTERVAL`` s
        until the PS1 count increments or no-change / hard timeout fires.

    All timing-sensitive work (PTY buffering, line discipline, key
    interpretation) lives inside the container; the host issues only
    stateless ``docker exec tmux ...`` invocations. This sidesteps the
    PTY-over-docker-exec buffering that broke our previous PS1-sentinel
    detector for multi-statement input.

    Multi-statement input is rejected at the API layer (bashlex AST split)
    so each ``send-keys`` produces exactly one new prompt — making the
    PS1-count delta a reliable completion signal.
    """

    POLL_INTERVAL = 0.5
    HISTORY_LIMIT = 10_000
    PS1 = CmdOutputMetadata.to_ps1_prompt()

    def __init__(
        self,
        container_name: str,
        work_dir: str = "/testbed",
        docker_host: str | None = None,
    ) -> None:
        self.container_name = container_name
        self.work_dir = work_dir
        self.docker_host = docker_host
        self._tmux_lock = asyncio.Lock()
        self._tmux_initialized = False
        self._tmux_session = f"agent-{uuid.uuid4().hex[:12]}"
        self._runtime_cwd: str | None = None
        self.live_bash_log_path: str | None = None
        self._initial_head: str | None = None

    def _subprocess_env(self) -> dict[str, str] | None:
        if not self.docker_host:
            return None
        env = os.environ.copy()
        env["DOCKER_HOST"] = self.docker_host
        return env

    async def exec(self, command: list[str], timeout: int = 180, work_dir: str | None = None) -> ExecResult:
        wd = work_dir or self.work_dir
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", "-w", wd, self.container_name, *command,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=self._subprocess_env(),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return ExecResult(
                success=proc.returncode == 0,
                returncode=proc.returncode or 0,
                # errors="replace": git diff of binary-ish repos (bitstring,
                # dulwich) can emit non-UTF-8 bytes; a hard decode here would
                # crash get_patch and silently drop the whole patch.
                stdout=stdout.decode(errors="replace") if stdout else "",
                stderr=stderr.decode(errors="replace") if stderr else "",
            )
        except asyncio.TimeoutError:
            return ExecResult(False, -1, "", f"Timed out after {timeout}s")
        except Exception as e:
            return ExecResult(False, -1, "", str(e))

    async def exec_shell(self, command: str, timeout: int = 180, work_dir: str | None = None) -> ExecResult:
        return await self.exec(["bash", "-c", command], timeout=timeout, work_dir=work_dir)


    async def _docker_tmux(
        self, *tmux_args: str, timeout: float = 10.0,
    ) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", self.container_name, "tmux", *tmux_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._subprocess_env(),
        )
        try:
            out_b, err_b = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return -1, ""
        rc = proc.returncode or 0
        out = out_b.decode("utf-8", errors="replace")
        if rc != 0 and err_b:
            logger.debug(
                "docker exec tmux %s rc=%s stderr=%s",
                " ".join(tmux_args), rc, err_b.decode(errors="replace"),
            )
        return rc, out

    async def _ensure_tmux_installed(self) -> None:
        probe = await self.exec(["bash", "-c", "command -v tmux"], timeout=10)
        if probe.success:
            return
        logger.info(
            "tmux not present in %s — installing (apt-get update + install -y tmux)",
            self.container_name,
        )
        install = await self.exec(
            ["bash", "-c",
             "apt-get update -qq >/dev/null 2>&1 && "
             "apt-get install -qq -y tmux >/dev/null 2>&1"],
            timeout=120,
        )
        if not install.success:
            logger.warning(
                "tmux install failed quietly; retrying with stderr surfaced"
            )
            install = await self.exec(
                ["bash", "-c", "apt-get update && apt-get install -y tmux"],
                timeout=180,
            )
        if not install.success:
            raise RuntimeError(
                f"Could not install tmux inside {self.container_name}: "
                f"{install.stderr or install.stdout}"
            )

    async def _ensure_tmux_session(self) -> None:
        if self._tmux_initialized:
            return
        await self._ensure_tmux_installed()
        rc, _ = await self._docker_tmux(
            "has-session", "-t", self._tmux_session,
        )
        if rc == 0:
            self._tmux_initialized = True
            return
        rc, _ = await self._docker_tmux(
            "new-session", "-d", "-s", self._tmux_session,
            "-x", "240", "-y", "50",
            "-c", self.work_dir,
            "/bin/bash",
        )
        if rc != 0:
            raise RuntimeError(
                f"tmux new-session failed for {self._tmux_session} in "
                f"container {self.container_name} (tmux installed but session start failed)"
            )
        await self._docker_tmux(
            "set-option", "-g", "history-limit", str(self.HISTORY_LIMIT),
        )
        init_cmd = (
            "export TERM=dumb; "
            "export PAGER=cat; "
            "export GIT_PAGER=cat; "
            "export MANPAGER=cat; "
            "export VIRTUAL_ENV_DISABLE_PROMPT=1; "
            "git config --global core.pager cat 2>/dev/null; "
            f"export PROMPT_COMMAND='export PS1=\"{self.PS1}\"'; "
            "export PS2=''"
        )
        await self._tmux_send_keys(init_cmd, enter=True)
        deadline = time.time() + 10.0
        while time.time() < deadline:
            await asyncio.sleep(0.2)
            content = await self._tmux_capture_pane()
            if CmdOutputMetadata.matches_ps1_metadata(content):
                self._tmux_initialized = True
                return
        logger.warning(
            "tmux session %s started but no PS1 sentinel observed in 10s; "
            "continuing — first command may include init noise",
            self._tmux_session,
        )
        self._tmux_initialized = True

    async def _tmux_send_keys(
        self, text: str, *, enter: bool, key: bool = False,
    ) -> None:
        """Send keystrokes to the tmux pane.

        Args:
            text: Either literal command text (``key=False``, default) or
                a tmux key name like ``Enter`` / ``C-c`` / ``C-z``
                (``key=True``). Literal mode passes ``-l`` so newlines
                and other bytes flow through verbatim instead of being
                parsed as tmux key names.
            enter: After ``text``, also send a separate ``Enter`` key.
                Use ``True`` on the final keystroke of a command body.
        """
        if key:
            await self._docker_tmux(
                "send-keys", "-t", self._tmux_session, text,
            )
        else:
            await self._docker_tmux(
                "send-keys", "-t", self._tmux_session, "-l", text,
            )
        if enter:
            await self._docker_tmux(
                "send-keys", "-t", self._tmux_session, "Enter",
            )

    async def _tmux_capture_pane(self) -> str:
        rc, out = await self._docker_tmux(
            "capture-pane", "-t", self._tmux_session, "-J", "-pS", "-",
            timeout=15.0,
        )
        if rc != 0:
            return ""
        return "\n".join(line.rstrip() for line in out.split("\n"))

    @staticmethod
    def _slice_command_output(
        pane: str,
        ps1_matches: list[re.Match],
        initial_ps1_count: int,
    ) -> str:
        if not ps1_matches:
            return pane
        if initial_ps1_count == 0:
            return pane[: ps1_matches[-1].start()]
        prior_idx = initial_ps1_count - 1
        if prior_idx >= len(ps1_matches) - 1:
            return pane[: ps1_matches[-1].start()]
        return pane[
            ps1_matches[prior_idx].end() + 1 : ps1_matches[-1].start()
        ]

    @staticmethod
    def _strip_command_echo(output: str, command: str) -> str:
        cmd_norm = "\n".join(line.rstrip() for line in command.split("\n"))
        out = output.lstrip("\n")
        cmd_norm = cmd_norm.lstrip("\n")
        if out.startswith(cmd_norm):
            out = out[len(cmd_norm):]
        return out.lstrip("\n").rstrip()

    async def exec_in_persistent_shell(
        self, command: str, timeout: int = 180,
    ) -> ExecResult:
        async with self._tmux_lock:
            await self._ensure_tmux_session()

            try:
                splited = split_bash_commands(command)
            except Exception as e:
                logger.debug(
                    "split_bash_commands failed (%s); passing through", e,
                )
                splited = [command]
            if len(splited) > 1:
                listing = "\n".join(
                    f"  ({i + 1}) {c}" for i, c in enumerate(splited)
                )
                msg = (
                    "ERROR: Cannot execute multiple commands at once.\n"
                    "Please run each command separately, OR chain them via "
                    "&& / ; / |.\nProvided commands:\n" + listing
                )
                return ExecResult(False, -1, "", msg)

            if not command.strip():
                return ExecResult(True, 0, "", "")

            # container (verbatim — no escaping needed since we're not
            cmd_id = uuid.uuid4().hex[:12]
            cmd_file = f"/tmp/_agent_cmd_{cmd_id}.sh"
            wr = await self.write_file(cmd_file, command)
            if not wr.success:
                return ExecResult(
                    False, -1, "",
                    f"Failed to stage command file {cmd_file}: {wr.stderr or wr.stdout}",
                )

            pane_cmd = (
                f". {cmd_file}; __rc=$?; rm -f {cmd_file}; (exit $__rc)"
            )

            initial_pane = await self._tmux_capture_pane()
            initial_matches = CmdOutputMetadata.matches_ps1_metadata(initial_pane)
            initial_ps1_count = len(initial_matches)

            await self._tmux_send_keys(pane_cmd, enter=True)

            start = time.time()
            while True:
                await asyncio.sleep(self.POLL_INTERVAL)
                cur_pane = await self._tmux_capture_pane()
                ps1_matches = CmdOutputMetadata.matches_ps1_metadata(cur_pane)
                cur_count = len(ps1_matches)

                if (
                    cur_count > initial_ps1_count
                    or cur_pane.rstrip().endswith(CMD_OUTPUT_PS1_END.rstrip())
                ):
                    result = self._extract_completed(
                        pane_cmd, cur_pane, ps1_matches,
                        initial_ps1_count,
                    )
                    await self._post_command_cleanup()
                    self._log_bash_event(
                        started_at=start,
                        ended_at=time.time(),
                        user_command=command,
                        pane_command=pane_cmd,
                        raw_pane=cur_pane,
                        result=result,
                        outcome="completed",
                    )
                    return result

                if timeout and time.time() - start >= timeout:
                    final_pane, final_matches = await self._kill_foreground(
                        initial_ps1_count,
                    )
                    result = self._extract_timeout(
                        pane_cmd, final_pane, final_matches,
                        initial_ps1_count,
                        budget=float(timeout),
                    )
                    self._log_bash_event(
                        started_at=start,
                        ended_at=time.time(),
                        user_command=command,
                        pane_command=pane_cmd,
                        raw_pane=final_pane,
                        result=result,
                        outcome="hard_timeout",
                    )
                    return result

    async def _post_command_cleanup(self) -> None:
        await self._docker_tmux(
            "clear-history", "-t", self._tmux_session,
        )

    def _log_bash_event(
        self,
        *,
        started_at: float,
        ended_at: float,
        user_command: str,
        pane_command: str,
        raw_pane: str,
        result: ExecResult,
        outcome: str,
    ) -> None:
        if not self.live_bash_log_path:
            return
        rec = {
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": round(ended_at - started_at, 4),
            "outcome": outcome,
            "container": self.container_name,
            "tmux_session": self._tmux_session,
            "work_dir": self.work_dir,
            "runtime_cwd": self._runtime_cwd,
            "user_command": user_command,
            "pane_command": pane_command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "raw_pane": raw_pane,
        }
        try:
            with open(self.live_bash_log_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
                f.flush()
        except Exception as e:
            logger.debug("live_bash_log append failed: %s", e)

    async def _wait_for_new_prompt(
        self, baseline_count: int, deadline: float,
    ) -> tuple[bool, str, list[re.Match]]:
        while time.time() < deadline:
            await asyncio.sleep(0.2)
            pane = await self._tmux_capture_pane()
            matches = CmdOutputMetadata.matches_ps1_metadata(pane)
            if len(matches) > baseline_count:
                return True, pane, matches
        pane = await self._tmux_capture_pane()
        matches = CmdOutputMetadata.matches_ps1_metadata(pane)
        return False, pane, matches

    async def _kill_foreground(
        self, baseline_count: int,
    ) -> tuple[str, list[re.Match]]:
        await self._tmux_send_keys("C-c", enter=False, key=True)
        ok, pane, matches = await self._wait_for_new_prompt(
            baseline_count, time.time() + 1.5,
        )
        if ok:
            await self._post_command_cleanup()
            return pane, matches
        await self._tmux_send_keys("C-c", enter=False, key=True)
        ok, pane, matches = await self._wait_for_new_prompt(
            baseline_count, time.time() + 1.5,
        )
        if ok:
            await self._post_command_cleanup()
            return pane, matches
        await self._tmux_send_keys("C-z", enter=False, key=True)
        await asyncio.sleep(0.3)
        await self._tmux_send_keys("kill -9 %1 2>/dev/null", enter=True)
        ok, pane, matches = await self._wait_for_new_prompt(
            baseline_count, time.time() + 2.0,
        )
        if not ok:
            logger.warning(
                "Foreground process did not exit after SIGINT×2 + SIGKILL "
                "in tmux session %s — pane may still be poisoned",
                self._tmux_session,
            )
        await self._post_command_cleanup()
        return pane, matches

    def _extract_completed(
        self,
        command: str,
        pane: str,
        ps1_matches: list[re.Match],
        initial_ps1_count: int,
    ) -> ExecResult:
        if not ps1_matches:
            return ExecResult(
                False, -1, pane,
                "PS1 metadata block missing — capture-pane returned malformed content",
            )
        raw = self._slice_command_output(pane, ps1_matches, initial_ps1_count)
        meta = CmdOutputMetadata.from_ps1_match(ps1_matches[-1])
        try:
            rc = int(meta.exit_code) if meta.exit_code is not None else -1
        except (TypeError, ValueError):
            rc = -1
        if meta.working_dir:
            self._runtime_cwd = meta.working_dir
        output = self._strip_command_echo(raw, command)
        return ExecResult(
            success=(rc == 0), returncode=rc, stdout=output, stderr="",
        )

    def _extract_timeout(
        self,
        command: str,
        pane: str,
        ps1_matches: list[re.Match],
        initial_ps1_count: int,
        *,
        budget: float,
    ) -> ExecResult:
        raw = self._slice_command_output(pane, ps1_matches, initial_ps1_count)
        output = self._strip_command_echo(raw, command)
        msg = (
            f"\n[Command timed out after {budget}s; killed via SIGINT × 2 "
            f"(then SIGKILL via 'kill -9 %1' if still running).]"
        )
        return ExecResult(False, 124, output, msg)

    async def close_persistent_shell(self) -> None:
        if not self._tmux_initialized:
            return
        await self._docker_tmux(
            "kill-session", "-t", self._tmux_session, timeout=5.0,
        )
        self._tmux_initialized = False

    # Upstream SWE-bench-Live (evaluation/evaluation.py:87) handles the ~1%
    _NESTED_REPO_CD = (
        "[ -d .git ] || { "
        "g=$(find . -maxdepth 2 -mindepth 2 -type d -name .git -print -quit); "
        "[ -n \"$g\" ] && cd \"${g%/.git}\"; "
        "} ; "
    )

    async def snapshot_initial_head(self) -> str:
        r = await self.exec_shell(
            f"cd {self.work_dir}; {self._NESTED_REPO_CD} git rev-parse HEAD"
        )
        sha = (r.stdout or "").strip() if r.success else ""
        if sha:
            self._initial_head = sha
        else:
            logger.warning("snapshot_initial_head: git rev-parse HEAD failed: %s", r.stderr)
        return sha

    async def get_patch(self) -> str:
        if not self._initial_head:
            logger.warning(
                "get_patch: _initial_head not set — falling back to diff vs HEAD "
                "(will miss committed changes)"
            )
            base = "HEAD"
        else:
            base = self._initial_head

        excludes = " ".join([
            "':(exclude)*.bak'",       "':(exclude)**/*.bak'",
            "':(exclude)*.pyc'",       "':(exclude)**/*.pyc'",
            "':(exclude)*.pyo'",       "':(exclude)**/*.pyo'",
            "':(exclude)__pycache__'", "':(exclude)**/__pycache__'",
            "':(exclude)**/__pycache__/**'",
            "':(exclude)*.tmp'",       "':(exclude)**/*.tmp'",
            "':(exclude)*.swp'",       "':(exclude)**/*.swp'",
            "':(exclude)*.swo'",       "':(exclude)**/*.swo'",
            "':(exclude)*.orig'",      "':(exclude)**/*.orig'",
            "':(exclude)*.rej'",       "':(exclude)**/*.rej'",
            "':(exclude).DS_Store'",   "':(exclude)**/.DS_Store'",
        ])
        add = await self.exec_shell(
            f"cd {self.work_dir}; {self._NESTED_REPO_CD} git add -A"
        )
        result = await self.exec_shell(
            f"cd {self.work_dir}; {self._NESTED_REPO_CD} "
            f"git --no-pager diff {base} --text -- . {excludes}"
        )
        if not result.success or not (result.stdout or "").strip():
            logger.warning(
                "get_patch produced empty result for container=%s "
                "work_dir=%s base=%s — git add: rc=%s stderr=%s; "
                "git diff: rc=%s stdout_len=%d stderr=%s",
                self.container_name, self.work_dir, base,
                add.returncode, (add.stderr or "")[:200],
                result.returncode, len(result.stdout or ""),
                (result.stderr or "")[:300],
            )
        return result.stdout if result.success else ""

    async def write_file(self, path: str, content: str) -> "ExecResult":
        quoted = "'" + path.replace("'", "'\\''") + "'"
        encoded = base64.b64encode(content.encode()).decode()
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", "-i", "-w", self.work_dir,
                self.container_name,
                "bash", "-c", f"base64 -d > {quoted}",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._subprocess_env(),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=encoded.encode()),
                timeout=180,
            )
            return ExecResult(
                success=proc.returncode == 0,
                returncode=proc.returncode or 0,
                stdout=stdout.decode(errors="replace") if stdout else "",
                stderr=stderr.decode(errors="replace") if stderr else "",
            )
        except asyncio.TimeoutError:
            return ExecResult(False, -1, "", "write_file timed out after 180s")
        except Exception as e:
            return ExecResult(False, -1, "", f"write_file: {e}")

    async def read_file(self, path: str) -> str:
        result = await self.exec(["cat", path])
        return result.stdout if result.success else ""

    async def file_exists(self, path: str) -> bool:
        result = await self.exec(["test", "-f", path])
        return result.success

    async def dir_exists(self, path: str) -> bool:
        result = await self.exec(["test", "-d", path])
        return result.success


def _docker_env(docker_host: str | None) -> dict[str, str] | None:
    if not docker_host:
        return None
    env = os.environ.copy()
    env["DOCKER_HOST"] = docker_host
    return env


async def find_sandbox_container(
    session_name: str,
    docker_host: str | None = None,
) -> str | None:
    proc = await asyncio.create_subprocess_exec(
        "docker", "ps", "-a",
        "--filter", "name=openhands-runtime-",
        "--format", "{{.Names}}",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env=_docker_env(docker_host),
    )
    stdout, _ = await proc.communicate()
    containers = [c for c in stdout.decode().strip().split("\n") if c]

    for c in containers:
        if session_name in c:
            logger.info("Found sandbox container: %s (matched session %s)", c, session_name)
            return c

    logger.warning(
        "No openhands-runtime container matched session %s (candidates: %s)",
        session_name, containers or "none",
    )
    return None


async def extract_patch_from_sandbox(
    session_name: str,
    work_dir: str = "/testbed",
    docker_host: str | None = None,
) -> str:
    try:
        container = await find_sandbox_container(session_name, docker_host=docker_host)
        if not container:
            print(f"[patch] No sandbox container found for session {session_name}")
            return ""

        await asyncio.create_subprocess_exec(
            "docker", "start", container,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=_docker_env(docker_host),
        )
        await asyncio.sleep(1)

        diff_cmds = [
            f"cd {work_dir} && git add -A && git diff --cached HEAD",
            f"cd {work_dir} && git diff HEAD",
            f"cd {work_dir} && git diff HEAD~1 HEAD",
            f"cd {work_dir} && git log --oneline -10 | tail -1 | cut -d' ' -f1 | xargs -I{{}} git diff {{}} HEAD",
        ]
        for diff_cmd in diff_cmds:
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", container, "bash", "-c", diff_cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=_docker_env(docker_host),
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0 and stdout.decode().strip():
                print(f"[patch] Extracted from sandbox ({len(stdout.decode())} chars)")
                return stdout.decode()
        print(f"[patch] No diff found in sandbox container {container}")
        return ""
    except Exception as e:
        logger.error("Error extracting patch: %s", e)
        return ""


async def apply_patch_to_container(
    container_name: str,
    patch: str,
    work_dir: str = "/testbed",
    docker_host: str | None = None,
) -> bool:
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(patch)
            patch_file = f.name
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "cp", patch_file, f"{container_name}:/tmp/patch.diff",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=_docker_env(docker_host),
            )
            await proc.communicate()

            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", "-w", work_dir, container_name,
                "bash", "-c", "git apply /tmp/patch.diff && rm /tmp/patch.diff",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=_docker_env(docker_host),
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                return True

            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", "-w", work_dir, container_name,
                "bash", "-c", "git apply --3way /tmp/patch.diff; rm -f /tmp/patch.diff",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=_docker_env(docker_host),
            )
            await proc.communicate()
            return proc.returncode == 0
        finally:
            os.unlink(patch_file)
    except Exception as e:
        logger.error("Error applying patch: %s", e)
        return False


async def copy_sandbox_to_container(
    sandbox: str, target: str, target_dir: str = "/app",
    sandbox_dir: str = "/testbed",
    docker_host: str | None = None,
) -> bool:
    try:
        await asyncio.create_subprocess_exec(
            "docker", "start", sandbox,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=_docker_env(docker_host),
        )
        await asyncio.sleep(1)

        tmp_path = f"/tmp/_sandbox_copy_{os.getpid()}"

        proc_export = await asyncio.create_subprocess_exec(
            "docker", "cp", f"{sandbox}:{sandbox_dir}/.", tmp_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=_docker_env(docker_host),
        )
        await proc_export.communicate()

        proc_import = await asyncio.create_subprocess_exec(
            "docker", "cp", f"{tmp_path}/.", f"{target}:{target_dir}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=_docker_env(docker_host),
        )
        await proc_import.communicate()

        await asyncio.create_subprocess_exec(
            "rm", "-rf", tmp_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )

        return proc_import.returncode == 0
    except Exception as e:
        logger.error("Error copying sandbox files: %s", e)
        return False


async def cleanup_sandbox(
    session_name: str, docker_host: str | None = None,
) -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "ps", "-a",
            "--filter", "name=openhands-runtime-",
            "--format", "{{.Names}}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=_docker_env(docker_host),
        )
        stdout, _ = await proc.communicate()
        for c in stdout.decode().strip().split("\n"):
            if c and session_name in c:
                await asyncio.create_subprocess_exec(
                    "docker", "rm", "-f", c,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    env=_docker_env(docker_host),
                )
    except Exception:
        pass
