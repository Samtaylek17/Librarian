"""Shared base class for MAS runners.

Collapses the boilerplate that every runner repeated: config-field storage
in ``__init__``, ``setup()``, ``_make_runtime_config()``, the caveman
prose-style overlay, and the ``last_n_observations`` activation print.

A subclass supplies two class attributes — ``CONFIG`` (a dataclass whose
fields ARE the runner's knobs + defaults) and ``LABEL`` (the print
prefix) — and a ``run_with_executor``. Constructor defaults come straight
from ``CONFIG`` fields, so the dataclass is the single source of truth for
both "what can I tune" and "what's the default".
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable

from eval_pipeline.agents.base import AgentRunner
from eval_pipeline.agents.bases.openhands_base import build_openhands_agent
from eval_pipeline.agents.llm.sampling import pop_sampling_kwargs
from eval_pipeline.agents.mas._caveman import apply_caveman
from eval_pipeline.agents.runtime import RuntimeConfig
from eval_pipeline.utils.docker import DockerExecutor

logger = logging.getLogger(__name__)


class MASRunnerBase(AgentRunner):
    # ── subclass contract ────────────────────────────────────────────
    CONFIG: type                 # dataclass: fields = knobs + defaults
    LABEL: str = "MAS"           # print prefix family name
    CAVEMAN_ROLES: tuple[str, ...] = ()   # roles named in the caveman print
    USE_BASE_NAME_IN_LABEL: bool = True
    # cached-token block size: False ⇒ hard-coded 16 (the MAS overlays'
    # historical behaviour), True ⇒ derived from the model (librarian/vanilla).
    BLOCK_SIZE_FROM_MODEL: bool = False

    def __init__(
        self,
        *,
        base_agent_builder: Callable = build_openhands_agent,
        base_name: str = "",
        **kwargs: Any,
    ) -> None:
        self.base_agent_builder = base_agent_builder
        self.base_name = base_name
        for f in dataclasses.fields(type(self).CONFIG):
            setattr(self, f.name, kwargs.pop(f.name, getattr(type(self).CONFIG, f.name)))
        self._post_init(kwargs)
        self.sampling_kwargs = pop_sampling_kwargs(kwargs, model=self.model)
        if kwargs:
            logger.debug("[%s] unused kwargs: %s", self.LABEL, list(kwargs))
        self.docker: DockerExecutor | None = None
        self._docker: DockerExecutor | None = None
        self.call_log = None
        self._work_dir: str = "/testbed"

    def _post_init(self, kwargs: dict[str, Any]) -> None:
        """Hook: consume subclass-only constructor kwargs (e.g. memory
        strategy). Pop what you use from ``kwargs``; the leftover is logged
        as unused. Default drops the universally-accepted no-op extras."""
        kwargs.pop("runtime_image", None)

    # ── label / lifecycle ────────────────────────────────────────────
    @property
    def label(self) -> str:
        if self.USE_BASE_NAME_IN_LABEL and self.base_name:
            return f"{self.LABEL}/{self.base_name}"
        return self.LABEL

    async def setup(self, container_name: str, work_dir: str = "/testbed") -> bool:
        self.docker = DockerExecutor(
            container_name, work_dir, docker_host=self.docker_host,
        )
        self._docker = self.docker
        self._work_dir = work_dir
        await self.docker.snapshot_initial_head()
        print(f"[{self.label}] Ready (model={self.model})")
        return True

    @property
    def max_turns(self) -> int:
        mi = getattr(self, "max_iterations", None)
        return mi if mi is not None else getattr(self, "planner_max_turns", 1000)

    def _make_runtime_config(self) -> RuntimeConfig:
        return RuntimeConfig(
            run_timeout=self.run_timeout,
            max_turns=self.max_turns,
            max_nudges=self.max_nudges,
            max_observation_length=self.max_observation_length,
            bash_timeout=self.bash_timeout,
        )

    # ── prose-style / context overlays ───────────────────────────────
    def caveman_fn(self) -> Callable[[str], str]:
        """Return the caveman prose overlay (identity when disabled);
        print activation once over ``CAVEMAN_ROLES``."""
        if not self.caveman_level:
            return lambda s: s
        roles = " + ".join(self.CAVEMAN_ROLES) if self.CAVEMAN_ROLES else "all sub-agents"
        print(
            f"[{self.label}] Caveman overlay active "
            f"(level={self.caveman_level}) on {roles}"
        )
        return lambda s: apply_caveman(s, level=self.caveman_level)

    def note_last_n(self) -> None:
        if getattr(self, "last_n_observations", None) is not None:
            print(
                f"[{self.label}] last_n_observations overlay active: "
                f"{self.last_n_observations}"
            )
