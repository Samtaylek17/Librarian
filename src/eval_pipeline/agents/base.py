from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from eval_pipeline.agents.execution import ExecutionStrategy
    from eval_pipeline.agents.observability import LLMCallLog
    from eval_pipeline.agents.runtime import RuntimeConfig
    from eval_pipeline.utils.docker import DockerExecutor


@dataclass
class SubAgentSpec:

    name: str
    system_prompt: str
    invoke_template: str | None = None
    stop_at: list[str] = field(default_factory=lambda: ["submit"])
    base_tools: list[str] | None = None


@dataclass
class AgentResult:

    patch: str | None = None
    raw_output: Any = None
    exit_code: int = 0
    duration_seconds: float = 0.0
    error: str | None = None
    llm_calls: list[dict[str, Any]] = field(default_factory=list)


class AgentRunner(ABC):

    docker: "DockerExecutor | None" = None
    call_log: "LLMCallLog | None" = None

    docker_host: str | None = None

    @abstractmethod
    async def setup(self, container_name: str, work_dir: str = "/testbed") -> bool:
        ...

    async def run_with_executor(
        self,
        instruction: str,
        work_dir: str,
        *,
        executor: "ExecutionStrategy",
        runtime: "RuntimeConfig",
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement run_with_executor() "
            "or override run() directly."
        )

    def _make_runtime_config(self) -> "RuntimeConfig":
        from eval_pipeline.agents.runtime import RuntimeConfig
        return RuntimeConfig(
            run_timeout=getattr(self, "run_timeout", 7200),
            max_turns=getattr(self, "max_iterations", 1000),
            max_nudges=getattr(self, "max_nudges", 2),
            max_observation_length=getattr(self, "max_observation_length", 16000),
            bash_timeout=getattr(self, "bash_timeout", 1800),
        )

    async def run(
        self,
        container_name: str,
        instruction: str,
        work_dir: str,
    ) -> AgentResult:
        from eval_pipeline.agents.execution import StandardExecution
        from eval_pipeline.agents.runtime import (
            ContextWindowExceeded, EpisodeStuckAbort, StuckTerminate,
        )

        runtime = self._make_runtime_config()
        executor = StandardExecution(total_timeout=float(runtime.run_timeout))
        start = time.monotonic()
        error: str | None = None

        try:
            await self.run_with_executor(
                instruction, work_dir,
                executor=executor, runtime=runtime,
            )
        except asyncio.TimeoutError:
            error = f"run_timeout: exceeded {runtime.run_timeout}s"
        except ContextWindowExceeded as e:
            error = f"context_window_exceeded: {e}"
        except StuckTerminate as e:
            # Match upstream OpenHands: stuck → graceful terminal, commit
            error = f"stuck_terminate: {e}"
        except EpisodeStuckAbort as e:
            # Orchestrator re-delegated a wedged role past its budget —
            # end the episode gracefully (same terminal handling as stuck).
            error = f"stuck_terminate: episode aborted ({e})"
        except Exception as e:
            import traceback
            traceback.print_exc()
            error = f"{type(e).__name__}: {e}"

        patch = (
            await self.docker.get_patch() if self.docker is not None else None
        )
        duration = time.monotonic() - start
        if self.call_log is not None:
            llm_calls = self.call_log.to_dicts()
            totals = self.call_log.total()
        else:
            llm_calls = []
            totals = {}

        # NOTE: ``llm_calls`` is stored once, as the top-level
        # ``EpisodeData.llm_calls``. Do NOT embed it here too — that
        # doubled the saved episode size (the trajectory is ~50% of the
        # file). ``agent_raw_output`` keeps only the small aggregates.
        raw_output: dict = {"totals": totals}
        pool = getattr(self, "_librarian_pool", None)
        if pool is not None:
            try:
                raw_output["pool_stats"] = pool.stats()
            except Exception:
                pass

        return AgentResult(
            patch=patch or None,
            raw_output=raw_output,
            exit_code=0,
            duration_seconds=duration,
            llm_calls=llm_calls,
            error=error,
        )
