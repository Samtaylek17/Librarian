from __future__ import annotations

import argparse
import asyncio
import glob
import json
import re
import sys
from pathlib import Path

_EVAL_ERROR_MARKERS = [
    (re.compile(r"^Harness failed.*?(ConnectionError|Name or service not known)", re.DOTALL),
     "harness_dns_error"),
    (re.compile(r"^Harness failed.*?Max retries exceeded", re.DOTALL),
     "harness_network_error"),
    (re.compile(r"^Harness failed"), "harness_subprocess_error"),
    (re.compile(r"^Harness timed out"), "harness_timeout"),
    (re.compile(r"^Report not found"), "harness_no_report"),
    (re.compile(r"ConnectionError.*raw\.githubusercontent\.com", re.DOTALL),
     "harness_dns_error"),
]


def detect_eval_error(eval_raw_output: str) -> str | None:
    if not eval_raw_output:
        return None
    for pat, label in _EVAL_ERROR_MARKERS:
        if pat.search(eval_raw_output):
            return label
    return None


def find_candidates(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*Evaluator_*.json")):
        if not p.name.endswith(".json"):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if detect_eval_error(d.get("eval_raw_output") or ""):
            out.append(p)
    return out


def _resolve_benchmark_name(name: str) -> tuple[str, type]:
    from eval_pipeline.benchmarks import BENCHMARK_REGISTRY
    cls = BENCHMARK_REGISTRY.get(name)
    if cls is not None:
        return name, cls
    for k, v in BENCHMARK_REGISTRY.items():
        if v.__name__ == name:
            return k, v
    raise KeyError(
        f"Unknown benchmark {name!r}; known keys={list(BENCHMARK_REGISTRY)}"
    )


def _benchmark_for(name: str, split: str | None):
    key, cls = _resolve_benchmark_name(name)
    if key == "swe_bench_live" and split:
        return cls(split=split)
    return cls()


async def reevaluate_one(json_path: Path, *, apply: bool,
                         split_override: str | None = None,
                         test_timeout: int | None = None) -> dict:
    d = json.loads(json_path.read_text())
    task_id = d["task_id"]
    bench_name = d["benchmark"]
    bench_key, _ = _resolve_benchmark_name(bench_name)

    split = None
    if bench_key == "swe_bench_live":
        split = (split_override
                 or d.get("benchmark_split")
                 or (d.get("eval_metadata") or {}).get("split")
                 or "verified")

    patch = d.get("patch") or ""
    if not patch:
        return {"path": str(json_path), "skipped": "empty_patch"}

    bench = _benchmark_for(bench_name, split)
    if test_timeout is not None:
        bench.eval_config.test_timeout = test_timeout
    samples = bench.get_samples()
    sample = next((s for s in samples if s.id == task_id), None)
    if sample is None:
        return {"path": str(json_path), "skipped": f"sample_not_found:{task_id}"}

    print(f"  [reeval] {task_id} ({bench_name}{f'/{split}' if split else ''})...")

    try:
        await bench.setup_container(sample)
    except Exception as e:
        print(f"    setup_container failed (continuing): {type(e).__name__}: {e}")

    try:
        result = await bench.evaluate(sample, patch)
    except Exception as e:
        await bench.teardown(sample)
        return {"path": str(json_path),
                "error": f"{type(e).__name__}: {e}"}

    await bench.teardown(sample)

    new_passed = bool(result.passed)
    new_score = float(result.score)
    new_raw = result.raw_output or ""
    new_meta = result.metadata or {}

    summary = {
        "path": str(json_path),
        "task_id": task_id,
        "old_passed": d.get("passed"),
        "new_passed": new_passed,
        "new_score": new_score,
        "new_eval_error": detect_eval_error(new_raw),
    }

    if apply:
        d["passed"] = new_passed
        d["score"] = new_score
        d["eval_raw_output"] = new_raw
        d["eval_metadata"] = new_meta
        json_path.write_text(json.dumps(d, indent=2, default=str))
        summary["written"] = True
    else:
        summary["written"] = False

    return summary


async def main_async(args: argparse.Namespace) -> int:
    candidates: list[Path] = []
    if args.path:
        for p in args.path:
            if "*" in p:
                candidates.extend(Path(x) for x in glob.glob(p))
            else:
                candidates.append(Path(p))
    elif args.matrix_dir:
        candidates = find_candidates(Path(args.matrix_dir))
    else:
        print("Either --matrix-dir or --path is required.", file=sys.stderr)
        return 2

    if not candidates:
        print("No eval-error episodes found.")
        return 0

    print(f"Found {len(candidates)} episode(s) with eval-side errors:")
    for p in candidates:
        d = json.loads(p.read_text())
        label = detect_eval_error(d.get("eval_raw_output") or "")
        agent = p.parent.name
        task = p.parent.parent.name
        print(f"  {task} / {agent}  ({label})")

    if not args.apply:
        print("\n(dry run — pass --apply to actually re-evaluate)")
        return 0

    print(f"\nRe-evaluating {len(candidates)} episode(s)...")
    results: list[dict] = []
    sem = asyncio.Semaphore(max(1, args.parallel))

    async def runner(p: Path) -> None:
        async with sem:
            r = await reevaluate_one(p, apply=True, split_override=args.split,
                                     test_timeout=args.test_timeout)
            results.append(r)
            print(f"    -> {r}", flush=True)

    await asyncio.gather(*[runner(p) for p in candidates])

    print("\n=== Re-evaluation summary ===")
    n_pass = sum(1 for r in results if r.get("new_passed"))
    n_fail = sum(1 for r in results
                 if r.get("written") and r.get("new_passed") is False
                 and not r.get("new_eval_error"))
    n_still_err = sum(1 for r in results if r.get("new_eval_error"))
    n_err = sum(1 for r in results if r.get("error"))
    print(f"  pass: {n_pass}")
    print(f"  fail: {n_fail}")
    print(f"  still eval_error: {n_still_err}")
    print(f"  re-eval crashed: {n_err}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix-dir", help="Walk this dir for episode JSONs.")
    p.add_argument("--path", nargs="*", help="Specific episode JSON path(s); supports glob.")
    p.add_argument("--apply", action="store_true",
                   help="Actually re-evaluate and write results back.")
    p.add_argument("--parallel", type=int, default=2,
                   help="Concurrent re-evaluations (default 2; raise carefully — "
                        "the swebench harness spins up its own docker containers).")
    p.add_argument("--split", default=None,
                   help="SWE-bench Live split override (verified | lite | test | "
                        "full). Use when the episode JSON does not record the "
                        "split it was scored under.")
    p.add_argument("--test-timeout", type=int, default=None,
                   help="Override swebench harness --timeout (per-test cap "
                        "inside the eval container, in seconds). Default uses "
                        "EvalConfig.test_timeout (typically 5400 = 90 min).")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
