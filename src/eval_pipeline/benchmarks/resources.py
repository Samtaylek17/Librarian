"""Per-container resource limits — single source of truth.

Without limits, both the agent-action container (``setup_container``) and the
patch-evaluation container (swebench harness) inherit Docker's *unbounded*
defaults (unlimited CPU/memory/PIDs, 64 MB ``/dev/shm``). Results then depend
on host size: a 256-core/1 TB box lets the agent compile and run tests far
faster (and lets memory-hungry suites pass) than a 32-core/64 GB box. That is
unfair across hardware and irreproducible.

The major harnesses do NOT impose limits by default — SWE-bench, SWE-agent /
swe-rex, and OpenHands all leave containers unbounded (OpenHands sets only
``--ulimit nofile=65536``; terminal-bench pins ``cpus: '1'`` per task). There
is no comprehensive convention to adopt, so we borrow the *mechanisms*
(swebench ``nano_cpus``, OpenHands' nofile ulimit, terminal-bench's CPU cap)
and define a fixed, host-independent policy applied identically to both
containers.

``ResourceLimits`` renders to three forms:
  - ``docker_run_args()``  — CLI flags for ``docker run`` (agent container)
  - ``create_kwargs()``    — docker-py kwargs for ``containers.create/run``
                             (eval container, via _swebench_resource_patch)
  - ``to_env()`` / ``from_env()`` — cross-process transport to the eval
                             subprocess, which can't see this object directly.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass
class ResourceLimits:
    """Fixed, host-independent caps applied to every benchmark container.

    Defaults are generous enough not to spuriously fail SWE-bench Verified
    builds/tests (peaks: a few GB RAM, compile-heavy C extensions), while
    still capping every axis so a big host and a small host behave the same.
    The point of fairness is that these numbers do NOT scale with the host.
    """

    enabled: bool = True
    cpus: float = 4.0              # CPU-core ceiling      (docker --cpus / nano_cpus)
    memory: str = "16g"           # hard memory cap       (docker --memory / mem_limit)
    memory_swap: str | None = "16g"  # == memory -> swap OFF: deterministic OOM, no
                                  #                     host-dependent swap thrash
    pids: int = 4096              # max processes         (fork-bomb / runaway-xdist guard)
    shm_size: str = "2g"          # /dev/shm (64 MB default starves pytest-xdist / torch)
    nofile: str = "65536:65536"   # open-fd soft:hard     (OpenHands precedent)

    # ── agent container: `docker run` CLI flags ──────────────────────────────
    def docker_run_args(self) -> list[str]:
        if not self.enabled:
            return []
        args: list[str] = []
        if self.cpus:
            args += ["--cpus", str(self.cpus)]
        if self.memory:
            args += ["--memory", self.memory]
            if self.memory_swap:
                args += ["--memory-swap", self.memory_swap]
        if self.pids:
            args += ["--pids-limit", str(self.pids)]
        if self.shm_size:
            args += ["--shm-size", self.shm_size]
        if self.nofile:
            args += ["--ulimit", f"nofile={self.nofile}"]
        return args

    # ── eval container: docker-py containers.create/run kwargs ───────────────
    def create_kwargs(self) -> dict:
        if not self.enabled:
            return {}
        from docker.types import Ulimit

        kw: dict = {}
        if self.cpus:
            kw["nano_cpus"] = int(float(self.cpus) * 1_000_000_000)
        if self.memory:
            kw["mem_limit"] = self.memory
            if self.memory_swap:
                kw["memswap_limit"] = self.memory_swap
        if self.pids:
            kw["pids_limit"] = int(self.pids)
        if self.shm_size:
            kw["shm_size"] = self.shm_size
        if self.nofile:
            soft, _, hard = self.nofile.partition(":")
            kw["ulimits"] = [
                Ulimit(name="nofile", soft=int(soft), hard=int(hard or soft))
            ]
        return kw

    # ── cross-process transport (eval subprocess) ────────────────────────────
    ENV_PREFIX = "EVAL_RLIMIT_"

    def to_env(self) -> dict[str, str]:
        p = self.ENV_PREFIX
        return {
            f"{p}ENABLED": "1" if self.enabled else "0",
            f"{p}CPUS": str(self.cpus or ""),
            f"{p}MEMORY": self.memory or "",
            f"{p}MEMSWAP": self.memory_swap or "",
            f"{p}PIDS": str(self.pids or ""),
            f"{p}SHM": self.shm_size or "",
            f"{p}NOFILE": self.nofile or "",
        }

    @classmethod
    def from_env(cls, env: dict | None = None) -> "ResourceLimits":
        env = os.environ if env is None else env
        p = cls.ENV_PREFIX
        if p + "ENABLED" not in env:
            # No wiring present (e.g. harness invoked directly) -> no-op.
            return cls(enabled=False)

        def g(k: str) -> str:
            return env.get(p + k, "")

        return cls(
            enabled=g("ENABLED") == "1",
            cpus=float(g("CPUS")) if g("CPUS") else 0.0,
            memory=g("MEMORY"),
            memory_swap=g("MEMSWAP") or None,
            pids=int(g("PIDS")) if g("PIDS") else 0,
            shm_size=g("SHM"),
            nofile=g("NOFILE"),
        )

    # ── recording / logging ──────────────────────────────────────────────────
    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        if not self.enabled:
            return "disabled (unbounded host resources)"
        return (
            f"cpus={self.cpus} mem={self.memory} swap={self.memory_swap} "
            f"pids={self.pids} shm={self.shm_size} nofile={self.nofile}"
        )
