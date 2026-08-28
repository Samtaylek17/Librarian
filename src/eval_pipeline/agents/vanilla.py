from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

from eval_pipeline.agents._sampling import pop_sampling_kwargs
from eval_pipeline.agents.base import AgentResult, AgentRunner, SubAgentSpec
from eval_pipeline.agents.bases.openhands_base import (
    StuckDetectionHooks,
    StuckRollbackHooks,
    _load_oh_system_prompt,
    build_openhands_agent,
)
from eval_pipeline.agents.bases.delegation import invoke_agent_logged
from eval_pipeline.agents.execution import ExecutionStrategy
from eval_pipeline.agents.runtime import AgentRegistry, CreateMode
from eval_pipeline.agents.observability import (
    LLMCallLog,
    setup_energy_telemetry,
)
from eval_pipeline.agents.llm import _ConfiguredModel
from eval_pipeline.agents.runtime import RuntimeConfig
from eval_pipeline.utils.docker import DockerExecutor


@dataclass
class VanillaConfig:

    model: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    run_timeout: int = 7200
    base_url: str = ""
    api_key: str = ""
    max_iterations: int = 1000
    max_observation_length: int = 16000
    max_nudges: int = 2
    bash_timeout: int = 1800
    seed: int | None = None
    stuck_rollback_turns: int | None = None
    max_rollbacks: int = 3


# Vendored OH system_prompt.j2 @ COMMIT_PIN.txt — verbatim; see
# third_party/openhands_vendored/prompts/system_prompt.j2. Loaded at
VANILLA_SYSTEM_PROMPT: str = _load_oh_system_prompt()


# OpenHands V1 official SWE-bench instance template (verbatim from
# Phase 2.1's ``conda activate testbed`` is part of the upstream template
# don't have conda); we keep the line as-is to stay verbatim.
_VANILLA_INSTANCE_TEMPLATE = """\
I have access to a python code repository in the directory {work_dir} . You can explore and modify files using the available tools. Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the {work_dir} directory to ensure the <issue_description> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets. Express in words any best practices or conventions in them.
   1.2 Hightlight message errors, method names, variables, file names, stack traces, and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Hightlight any best practices to take into account when testing and fixing the issue

Phase 2. RUNNING: install and run the tests on the repository
   2.1 Activate the environment by running
       ./opt/miniconda3/etc/profile.d/conda.sh ; conda activate testbed
   2.2 Follow the readme
   2.3 Install the environment and anything needed
   2.4 Iterate and figure out how to run the tests

Phase 3. EXPLORATION: find the files that are related to the problem and possible solutions
   3.1 Use `grep` to search for relevant methods, classes, keywords and error messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the methods and files to fix the issue and explain why.
   3.4 From the possible file locations, select the most likely location to fix the issue.

Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce and verify the issue.
   4.1 Look at existing test files in the repository to understand the test format/structure.
   4.2 Create a minimal reproduction script that reproduces the located issue.
   4.3 Run the reproduction script to confirm you are reproducing the issue.
   4.4 Adjust the reproduction script as necessary.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the test reproduces the issue.
   5.4 State clearly the best practices to take into account in the fix.
   5.5 State clearly how to fix the problem.

Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: Test your implementation thoroughly.
   7.1 Run your reproduction script to verify the fix works.
   7.2 Add edge cases to your test script to ensure comprehensive coverage.
   7.3 Run existing tests related to the modified code to ensure you haven't broken anything.

8. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit {base_commit}.
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any tests in the repository related to:
     8.2.1 The issue you are fixing
     8.2.2 The files you modified
     8.2.3 The functions you changed
   8.3 If any tests fail, revise your implementation until all tests pass

Be thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity."""


SUBAGENTS: dict[str, SubAgentSpec] = {
    "agent": SubAgentSpec(
        name="agent",
        system_prompt=VANILLA_SYSTEM_PROMPT,
        invoke_template=_VANILLA_INSTANCE_TEMPLATE,
        stop_at=["submit"],
    ),
}


class VanillaRunner(AgentRunner):

    def __init__(
        self,
        model: str = VanillaConfig.model,
        run_timeout: int = VanillaConfig.run_timeout,
        base_url: str = VanillaConfig.base_url,
        api_key: str = VanillaConfig.api_key,
        max_iterations: int = VanillaConfig.max_iterations,
        seed: int | None = None,
        max_nudges: int = VanillaConfig.max_nudges,
        max_observation_length: int = VanillaConfig.max_observation_length,
        bash_timeout: int = VanillaConfig.bash_timeout,
        stuck_rollback_turns: int | None = VanillaConfig.stuck_rollback_turns,
        max_rollbacks: int = VanillaConfig.max_rollbacks,
        runtime_image: str = "",
        **kwargs: Any,
    ) -> None:
        self.sampling_kwargs = pop_sampling_kwargs(kwargs, model=model)
        self.model = model
        self.run_timeout = run_timeout
        self.base_url = base_url
        self.api_key = api_key
        self.runtime_image = runtime_image
        self.max_iterations = max_iterations
        self.seed = seed
        self.max_nudges = max_nudges
        self.max_observation_length = max_observation_length
        self.bash_timeout = bash_timeout
        self.stuck_rollback_turns = stuck_rollback_turns
        self.max_rollbacks = max_rollbacks
        self.docker: DockerExecutor | None = None
        self.call_log: LLMCallLog | None = None
        self._docker: DockerExecutor | None = None
        self._work_dir: str = "/testbed"
        self._swebench_image: str = ""
        self._last_session: str = ""

    async def setup(self, container_name: str, work_dir: str = "/testbed") -> bool:
        self.docker = DockerExecutor(container_name, work_dir, docker_host=self.docker_host)
        self._docker = self.docker
        self._work_dir = work_dir
        await self.docker.snapshot_initial_head()

        import os as _os
        _docker_env = None
        if self.docker_host:
            _docker_env = _os.environ.copy()
            _docker_env["DOCKER_HOST"] = self.docker_host

        proc = await asyncio.create_subprocess_exec(
            "docker", "inspect", "--format={{.Config.Image}}", container_name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=_docker_env,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            self._swebench_image = stdout.decode().strip()

        print(f"[Vanilla] Ready (model={self.model})")
        return True

    async def run_with_executor(
        self,
        instruction: str,
        work_dir: str,
        *,
        executor: ExecutionStrategy,
        runtime: RuntimeConfig,
    ) -> None:
        from openai import AsyncOpenAI

        assert self.docker is not None, "Call setup() first"
        self.call_log = LLMCallLog(
            live_dump_path=(
                f"{self._live_log_dir}/live_calls.jsonl"
                if getattr(self, "_live_log_dir", None) else None
            ),
        )

        tokenizer = None
        block_size = 16
        try:
            from transformers import AutoTokenizer
            from eval_pipeline.utils.llm_proxy import LLMLoggingProxy
            tok_name = self.model.removeprefix("openai/")
            tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
            block_size = LLMLoggingProxy._get_block_size(tok_name)
        except Exception:
            pass

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "dummy")
        client = AsyncOpenAI(base_url=self.base_url or None, api_key=api_key, timeout=300.0, max_retries=0)
        live_http_log_path = (
            f"{self._live_log_dir}/live_http.jsonl"
            if getattr(self, "_live_log_dir", None) else None
        )
        llm_model = _ConfiguredModel(
            model=self.model,
            openai_client=client,
            seed=self.seed,
            tokenizer=tokenizer,
            live_http_log_path=live_http_log_path,
            **self.sampling_kwargs,
        )

        spec = SUBAGENTS["agent"]
        agent, _tools = build_openhands_agent(
            docker=self.docker,
            work_dir=work_dir,
            llm_model=llm_model,
            agent_name=spec.name,
            system_prompt=spec.system_prompt,
            stop_at_tool_names=spec.stop_at,
            max_observation_length=runtime.max_observation_length,
            bash_timeout=runtime.bash_timeout,
        )

        energy_url, idle_power_W = await setup_energy_telemetry(
            self.base_url, label="Vanilla",
        )

        if self.stuck_rollback_turns and self.stuck_rollback_turns > 0:
            print(
                f"[Vanilla] Stuck-rollback enabled: "
                f"rollback_turns={self.stuck_rollback_turns}, "
                f"max_rollbacks={self.max_rollbacks}"
            )

            def _stuck_factory():
                return StuckRollbackHooks(
                    rollback_turns=self.stuck_rollback_turns,
                    max_rollbacks=self.max_rollbacks,
                    max_nudges=runtime.max_nudges,
                )
        else:
            def _stuck_factory():
                return StuckDetectionHooks(max_nudges=runtime.max_nudges)

        if spec.invoke_template:
            first_user_msg = spec.invoke_template.format(
                work_dir=work_dir,
                problem_statement=instruction,
                base_commit=self.docker._initial_head or "HEAD",
            )
        else:
            first_user_msg = instruction

        self._registry = AgentRegistry()

        async def _execute(inst, hooks):
            return await executor.execute_one(
                agent=agent, input=first_user_msg,
                hooks=hooks, max_turns=runtime.max_turns,
            )

        print(f"[Vanilla] Running (max_turns={runtime.max_turns})...")
        # Top-level agent enters through the registry so it owns the
        # _CURRENT_AGENT context (requirement 1) and a stable agent_id.
        await invoke_agent_logged(
            registry=self._registry,
            call_log=self.call_log,
            agent_type="agent",
            agent_obj_factory=lambda: agent,
            execute=_execute,
            model=self.model,
            tokenizer=tokenizer,
            block_size=block_size,
            energy_url=energy_url,
            idle_power_W=idle_power_W,
            agent_id="agent_0",
            mode=CreateMode.CREATE,
            stuck_hooks_factory=_stuck_factory,
        )


OpenHandsBaseRunner = VanillaRunner
