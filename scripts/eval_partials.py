#!/usr/bin/env python
"""Phase-B eval-only dispatcher for orphan ``*.partial.json`` files.

Loads each partial, looks up its SWE-bench Sample, calls
``EpisodeRunner.run_eval_phase`` so we reuse the production phase-B
plumbing (extractors, container restore, retry, teardown, DOCKER_HOST
override). On success the final JSON is written and the partial is
removed.

DOCKER_HOST is taken from the environment (``ssh://worker-1`` etc.);
that env var is what swe_bench passes to ``docker.from_env()`` in its
subprocess.

Usage:
  DOCKER_HOST=ssh://worker-1 python scripts/eval_partials.py \\
    --log-dir logs/caveman_lastn_27b_500_6agents_seed0 \\
    --variants boad_lastn,hyperagent_lastn \\
    --parallel 8
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from eval_pipeline.benchmarks import BENCHMARK_REGISTRY
from eval_pipeline.extractors import default_extractors
from eval_pipeline.pipeline.episode import EpisodeRunner


def _list_partials(log_dir: Path, variants: list[str] | None) -> list[Path]:
    out: list[Path] = []
    for p in sorted(log_dir.rglob("*Evaluator_*.partial.json")):
        if variants:
            parent = p.parent.name
            for suffix in ("_seed0", "_seed1", "_seed2", "_seed3", "_seed4"):
                if parent.endswith(suffix):
                    parent = parent[: -len(suffix)]
                    break
            if parent not in variants:
                continue
        final = p.with_name(p.name[: -len(".partial.json")] + ".json")
        if final.exists():
            continue
        out.append(p)
    return out


def _resolve_benchmark(name: str):
    cls = BENCHMARK_REGISTRY.get(name)
    if cls is not None:
        return cls
    for k, v in BENCHMARK_REGISTRY.items():
        if v.__name__ == name:
            return v
    raise KeyError(f"unknown benchmark: {name!r}")


async def _eval_one(
    partial_path: Path,
    samples_by_id: dict,
    docker_host: str | None,
    sem: asyncio.Semaphore,
    counter: dict,
    total: int,
) -> str:
    async with sem:
        try:
            d = json.loads(partial_path.read_text())
            task_id = d.get("task_id")
            bench_name = d.get("benchmark") or "swe_bench"
            sample = samples_by_id.get(task_id)
            if sample is None:
                counter["skip"] += 1
                return f"SKIP {task_id} (not in benchmark samples)"

            bench_cls = _resolve_benchmark(bench_name)
            # Fresh evaluator per task (matches retro_eval_boad_upstream
            evaluator = bench_cls()
            if docker_host:
                evaluator.docker_host = docker_host

            stub_agent = SimpleNamespace(docker_host=None)
            runner = EpisodeRunner(
                agent=stub_agent,
                benchmark=evaluator,
                extractors=default_extractors(),
                log_dir=partial_path.parent,
                skip_teardown=False,
                docker_host=docker_host,
            )
            result = await runner.run_eval_phase(partial_path, sample)
            label = "PASS" if result.passed else "fail"
            counter["pass" if result.passed else "fail"] += 1
            counter["done"] += 1
            return (f"[{counter['done']}/{total}] {label} {task_id}  "
                    f"score={result.score}")
        except Exception as e:
            counter["err"] += 1
            counter["done"] += 1
            return (f"[{counter['done']}/{total}] ERR  {partial_path.name}: "
                    f"{type(e).__name__}: {e}")


async def _amain(log_dir: Path, variants: list[str] | None,
                 parallel: int, limit: int = 0):
    partials = _list_partials(log_dir, variants)
    print(f"orphan partials to evaluate: {len(partials)}", flush=True)
    if limit and limit > 0:
        partials = partials[:limit]
        print(f"  (limited to first {limit})", flush=True)
    if not partials:
        return

    first = json.loads(partials[0].read_text())
    bench_name = first.get("benchmark") or "swe_bench"
    print(f"benchmark: {bench_name}", flush=True)

    evaluator_for_samples = _resolve_benchmark(bench_name)()
    samples_by_id = {s.id: s for s in evaluator_for_samples.get_samples()}
    print(f"samples loaded: {len(samples_by_id)}", flush=True)

    docker_host = os.environ.get("DOCKER_HOST") or None
    if docker_host:
        print(f"DOCKER_HOST: {docker_host}", flush=True)

    sem = asyncio.Semaphore(parallel)
    counter = {"done": 0, "pass": 0, "fail": 0, "skip": 0, "err": 0}
    tasks = [
        asyncio.create_task(
            _eval_one(p, samples_by_id, docker_host, sem, counter, len(partials))
        )
        for p in partials
    ]
    for coro in asyncio.as_completed(tasks):
        msg = await coro
        print(msg, flush=True)

    print(
        f"\n=== summary ===\n"
        f"  total: {len(partials)}\n"
        f"  pass:  {counter['pass']}\n"
        f"  fail:  {counter['fail']}\n"
        f"  skip:  {counter['skip']}\n"
        f"  err:   {counter['err']}\n",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True,
                    help="Matrix log root; rglob's *.partial.json under it.")
    ap.add_argument("--variants", default=None,
                    help="Comma-separated variant names (= parent dir name with "
                         "_seedN stripped). Default: all variants found.")
    ap.add_argument("--parallel", type=int, default=4,
                    help="Concurrent evals (asyncio.Semaphore).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap on number of partials to process (sanity runs). "
                         "0 = all.")
    args = ap.parse_args()
    variants = (
        [v.strip() for v in args.variants.split(",") if v.strip()]
        if args.variants else None
    )
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        sys.exit(f"log_dir not found: {log_dir}")
    asyncio.run(_amain(log_dir, variants, args.parallel, args.limit))


if __name__ == "__main__":
    main()
