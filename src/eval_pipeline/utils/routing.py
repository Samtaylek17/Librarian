from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_DEFAULT_POOL_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "server_pool.yaml"
)


@dataclass(frozen=True)
class ServerSpec:
    name: str
    docker_host: str
    max_concurrent: int
    disk_free_gb: int = 0
    disk_budget_gb: int = 0
    prune_age: str = "4h"       # passed verbatim to ``--filter until=`` in prune script
    ram_gb: int = 0
    cpu_cores: int = 0
    cpu_max_ghz: float = 0.0


def _parse_age(s: str) -> int:
    m = re.match(r"^\s*(\d+)\s*([smhd])\s*$", s.lower())
    if not m:
        raise ValueError(f"invalid age expression: {s!r}")
    n = int(m.group(1))
    return n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def _load_servers(pool_path: Path | str | None) -> tuple[dict[str, ServerSpec], dict]:
    path = Path(pool_path) if pool_path else _DEFAULT_POOL_PATH
    with open(path) as f:
        spec = yaml.safe_load(f)
    raw = spec.get("servers") or {}
    servers = {
        name: ServerSpec(
            name=name,
            docker_host=meta["docker_host"],
            max_concurrent=int(meta.get("max_concurrent", 1)),
            disk_free_gb=int(meta.get("disk_free_gb", 0)),
            disk_budget_gb=int(meta.get("disk_budget_gb", 0)),
            prune_age=str(meta.get("prune_age", "4h")),
            ram_gb=int(meta.get("ram_gb", 0)),
            cpu_cores=int(meta.get("cpu_cores", 0)),
            cpu_max_ghz=float(meta.get("cpu_max_ghz", 0.0)),
        )
        for name, meta in raw.items()
    }
    return servers, spec


class DispatchRouter:

    def __init__(self, pool_path: Path | str | None = None):
        self.servers, self._spec = _load_servers(pool_path)
        self._enabled = bool(self._spec.get("routing", {}).get("enable", True))
        self._active: dict[str, int] = {n: 0 for n in self.servers}
        self._cond = threading.Condition()

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self.servers)

    @property
    def total_slots(self) -> int:
        return sum(s.max_concurrent for s in self.servers.values())

    def __repr__(self) -> str:
        return (
            f"DispatchRouter(enabled={self._enabled}, "
            f"servers={list(self.servers)}, total_slots={self.total_slots})"
        )

    def reserve(self, task_id: str | None = None) -> ServerSpec:
        with self._cond:
            while True:
                best: ServerSpec | None = None
                best_load = float("inf")
                for name in sorted(self.servers):
                    srv = self.servers[name]
                    if self._active[name] >= srv.max_concurrent:
                        continue
                    load = self._active[name] / srv.max_concurrent
                    if load < best_load:
                        best = srv
                        best_load = load
                if best is not None:
                    self._active[best.name] += 1
                    return best
                self._cond.wait()

    def release(self, server_name: str) -> None:
        with self._cond:
            if self._active.get(server_name, 0) > 0:
                self._active[server_name] -= 1
                self._cond.notify_all()

    def snapshot(self) -> dict[str, tuple[int, int]]:
        with self._cond:
            return {
                n: (self._active.get(n, 0), s.max_concurrent)
                for n, s in self.servers.items()
            }


class Router:

    def __init__(self, pool_path: Path | str | None = None):
        try:
            self.servers, self._spec = _load_servers(pool_path)
        except FileNotFoundError:
            self.servers = {}
            self._spec = {}

    @property
    def enabled(self) -> bool:
        return False

    def host_for(self, task_id: str) -> Optional[str]:
        return None

    def server_for(self, task_id: str) -> Optional[ServerSpec]:
        return None


_router_singleton: Router | None = None


def get_router() -> Router:
    global _router_singleton
    if _router_singleton is None:
        _router_singleton = Router()
    return _router_singleton
