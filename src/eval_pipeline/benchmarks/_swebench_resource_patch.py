"""Monkeypatch the docker SDK so swebench eval containers get the same
resource limits as the agent container.

The patch-evaluation harness runs as a *subprocess* — ``python -m
swebench.harness.run_evaluation`` — which imports the installed ``swebench``
(``docker_build.build_container``). That code calls
``client.containers.create(...)`` with no resource limits, so we cannot reach
it from the parent process. Instead the eval subprocess is bootstrapped to
``install()`` this patch first (see ``SWEBenchEvaluator.evaluate``), which
wraps ``ContainerCollection.create``/``run`` to merge in limits read from the
``EVAL_RLIMIT_*`` env vars (set by ``ResourceLimits.to_env``).

``setdefault`` semantics: we only inject a limit the harness didn't already
set, so any per-spec value (e.g. pylint ``nano_cpus=2e9``) wins over our
default. This is import-safe and version-agnostic — it patches the docker SDK,
not swebench internals.
"""
from __future__ import annotations

_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from eval_pipeline.benchmarks.resources import ResourceLimits

    limits = ResourceLimits.from_env()
    kwargs = limits.create_kwargs()
    if not kwargs:
        return  # disabled or nothing to set

    import docker.models.containers as _c

    orig_create = _c.ContainerCollection.create
    orig_run = _c.ContainerCollection.run

    def _merge(call_kwargs: dict) -> dict:
        for k, v in kwargs.items():
            call_kwargs.setdefault(k, v)
        return call_kwargs

    def create(self, *args, **kw):  # noqa: ANN001
        return orig_create(self, *args, **_merge(kw))

    def run(self, *args, **kw):  # noqa: ANN001
        return orig_run(self, *args, **_merge(kw))

    _c.ContainerCollection.create = create
    _c.ContainerCollection.run = run
    print(
        f"[resource-patch] eval containers limited: {limits.describe()}",
        flush=True,
    )
