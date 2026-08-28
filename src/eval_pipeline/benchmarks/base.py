from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from eval_pipeline.benchmarks.resources import ResourceLimits


@dataclass
class Sample:
    id: str
    instruction: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    score: float
    passed: bool
    raw_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:

    timeout: int = 1800
    test_timeout: int = 1800
    max_retries: int = 2
    retry_backoff_seconds: float = 5.0


_TRANSIENT_PREFIXES = (
    "Harness timed out",
    "Harness failed",
    "Report not found",
)

_TRANSIENT_SUBSTRINGS = (
    re.compile(r"ConnectionError"),
    re.compile(r"Name or service not known"),
    re.compile(r"Max retries exceeded"),
)


def is_transient_eval_failure(raw: str) -> bool:
    if not raw:
        return False
    if raw.startswith(_TRANSIENT_PREFIXES):
        return True
    return any(p.search(raw) for p in _TRANSIENT_SUBSTRINGS)


class InfraFailure(Exception):
    pass


class BenchmarkEvaluator(ABC):

    docker_host: str | None = None
    eval_config: EvalConfig = EvalConfig()
    # Per-container resource caps, applied identically to the agent-action
    # container and the patch-eval container (host-independent fairness).
    # Subclasses set their own instance in __init__; this is the fallback.
    resource_limits: ResourceLimits = ResourceLimits()

    @abstractmethod
    def get_samples(self, limit: int | None = None) -> list[Sample]:
        ...

    @abstractmethod
    async def setup_container(self, sample: Sample) -> tuple[str, str]:
        ...

    @abstractmethod
    async def evaluate(self, sample: Sample, patch: str | None) -> EvalResult:
        ...

    @abstractmethod
    async def teardown(self, sample: Sample) -> None:
        ...

    async def evaluate_with_retry(
        self, sample: Sample, patch: str | None,
    ) -> EvalResult:
        cfg = self.eval_config
        attempts: list[str] = []
        last_result: EvalResult | None = None

        for i in range(cfg.max_retries + 1):
            try:
                result = await self.evaluate(sample, patch)
            except asyncio.TimeoutError:
                result = EvalResult(
                    score=0.0, passed=False,
                    raw_output=f"Harness timed out (attempt {i+1})",
                )
            except Exception as e:
                if isinstance(e, InfraFailure):
                    raise
                result = EvalResult(
                    score=0.0, passed=False,
                    raw_output=f"Harness failed (attempt {i+1}): "
                               f"{type(e).__name__}: {e}",
                )

            last_result = result
            if not is_transient_eval_failure(result.raw_output):
                if attempts:
                    md = dict(result.metadata or {})
                    md["eval_attempts"] = attempts + [
                        f"attempt {i+1}: ok",
                    ]
                    result.metadata = md
                return result

            attempts.append(f"attempt {i+1}: {result.raw_output[:200]}")
            if i < cfg.max_retries:
                await asyncio.sleep(cfg.retry_backoff_seconds)

        assert last_result is not None
        md = dict(last_result.metadata or {})
        md["eval_attempts"] = attempts
        last_result.metadata = md
        return last_result
