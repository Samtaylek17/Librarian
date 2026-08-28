#!/usr/bin/env python3
"""
CLI entry point for the evaluation pipeline.

Commands:
  run         Run a single (agent, benchmark, task) episode
  experiment  Run a batch experiment from YAML config

Examples:
    python scripts/run.py run \
        --agent vanilla --benchmark swe_bench \
        --task-id django__django-14631 --base-url http://localhost:8000/v1

    python scripts/run.py experiment configs/experiments/example.yaml
"""

import asyncio
import argparse
import faulthandler
import signal
import sys
from pathlib import Path

faulthandler.register(signal.SIGUSR1, all_threads=True)


def _dump_asyncio_tasks(_signum, _frame):
    import asyncio
    import traceback
    sys.stderr.write("\n=== asyncio task dump ===\n")
    try:
        loop = asyncio.get_event_loop()
        tasks = asyncio.all_tasks(loop)
    except Exception as e:
        sys.stderr.write(f"  (could not enumerate tasks: {e})\n")
        return
    for i, task in enumerate(tasks):
        sys.stderr.write(f"\n--- Task[{i}] name={task.get_name()} done={task.done()} ---\n")
        try:
            coro = task.get_coro()
            depth = 0
            seen = set()
            while coro is not None and id(coro) not in seen and depth < 200:
                seen.add(id(coro))
                depth += 1
                frame = getattr(coro, 'cr_frame', None) or getattr(coro, 'gi_frame', None) or getattr(coro, 'ag_frame', None)
                if frame is not None:
                    sys.stderr.write(
                        f"  File \"{frame.f_code.co_filename}\", line {frame.f_lineno}, "
                        f"in {frame.f_code.co_name}\n"
                    )
                else:
                    sys.stderr.write(f"  (no frame on {type(coro).__name__})\n")
                nxt = getattr(coro, 'cr_await', None) or getattr(coro, 'gi_yieldfrom', None) or getattr(coro, 'ag_await', None)
                coro = nxt
        except Exception as e:
            sys.stderr.write(f"  (stack unavailable: {e})\n")
    sys.stderr.write("\n=== end asyncio task dump ===\n")
    sys.stderr.flush()


signal.signal(signal.SIGUSR2, _dump_asyncio_tasks)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

AVAILABLE_AGENTS = [
    "vanilla",
    "librarian", "librarian_caveman", "librarian_rag",
    "hyperagent", "hyperagent_caveman", "hyperagent_lastn",
    "hyperagent_librarian", "hyperagent_librarian_caveman",
    "hyperagent_librarian_rag",
    "hyperagent_librarian_nopersist", "hyperagent_librarian_nopointer",
    "hyperagent_librarian_noscope", "hyperagent_librarian_nofresh",
    "hyperagent_exec_librarian",
    "boad", "boad_caveman", "boad_lastn",
    "boad_librarian", "boad_librarian_caveman", "boad_librarian_rag",
    "boad_librarian_nopersist", "boad_librarian_nopointer",
    "boad_librarian_noscope", "boad_librarian_nofresh",
]
AVAILABLE_BENCHMARKS = ["swe_bench", "swe_bench_live"]


async def cmd_run(args):
    from eval_pipeline.agents import create_agent
    from eval_pipeline.benchmarks import create_benchmark
    from eval_pipeline.extractors import default_extractors
    from eval_pipeline.pipeline.episode import EpisodeRunner

    agent_kwargs = {}
    if args.model:
        agent_kwargs["model"] = args.model
    if args.base_url:
        agent_kwargs["base_url"] = args.base_url
    if args.api_key:
        agent_kwargs["api_key"] = args.api_key
    if args.agent_mode:
        agent_kwargs["agent_mode"] = args.agent_mode
    if args.run_timeout:
        agent_kwargs["run_timeout"] = args.run_timeout
    if args.seed is not None:
        agent_kwargs["seed"] = args.seed
    for name in ("temperature", "top_p", "top_k", "min_p",
                 "presence_penalty", "repetition_penalty",
                 "max_observation_length", "max_iterations",
                 "bash_timeout",
                 "stuck_rollback_turns", "max_rollbacks",
                 "memory_strategy",
                 "drop_below_threshold",
                 "retrieval_backend",
                 "retrieval_threshold",
                 "retrieval_max_pairs",
                 "retrieval_endpoint"):
        v = getattr(args, name, None)
        if v is not None:
            agent_kwargs[name] = v

    agent = create_agent(args.agent, **agent_kwargs)

    benchmark_kwargs = {}
    if getattr(args, "benchmark_split", None):
        benchmark_kwargs["split"] = args.benchmark_split
    benchmark = create_benchmark(args.benchmark, **benchmark_kwargs)

    samples = benchmark.get_samples()
    sample = next((s for s in samples if s.id == args.task_id), None)
    if not sample:
        print(f"Error: Task ID not found: {args.task_id}")
        return 1

    print("=" * 70)
    print(f"Agent: {args.agent} (model: {getattr(agent, 'model', 'N/A')})")
    print(f"Benchmark: {args.benchmark}")
    print(f"Task: {args.task_id}")
    print("=" * 70)
    print()

    runner = EpisodeRunner(
        agent=agent,
        benchmark=benchmark,
        extractors=default_extractors(),
        log_dir=args.log_dir,
        skip_teardown=args.skip_teardown,
        docker_host=args.docker_host,
    )

    if getattr(args, "eval_timeout", None) is not None:
        benchmark.eval_config.timeout = int(args.eval_timeout)
    if getattr(args, "eval_max_retries", None) is not None:
        benchmark.eval_config.max_retries = int(args.eval_max_retries)

    # Per-container resource limits (applied to agent + eval containers).
    rl = benchmark.resource_limits
    if getattr(args, "resource_limits", None) is False:
        rl.enabled = False
    if getattr(args, "limit_cpus", None) is not None:
        rl.cpus = args.limit_cpus
    if getattr(args, "limit_memory", None) is not None:
        rl.memory = args.limit_memory
    if getattr(args, "limit_memory_swap", None) is not None:
        rl.memory_swap = args.limit_memory_swap
    if getattr(args, "limit_pids", None) is not None:
        rl.pids = args.limit_pids
    if getattr(args, "limit_shm", None) is not None:
        rl.shm_size = args.limit_shm
    if getattr(args, "limit_nofile", None) is not None:
        rl.nofile = args.limit_nofile
    print(f"Resource limits: {rl.describe()}")

    from eval_pipeline.benchmarks.base import InfraFailure
    phase = getattr(args, "phase", "both")
    try:
        if phase == "both":
            result = await runner.run(sample)
        elif phase == "agent":
            runner.skip_teardown = True
            result, _container = await runner.run_agent_phase(sample)
        elif phase == "eval":
            partial_path = Path(args.partial_path)
            if not partial_path.exists():
                print(f"Error: partial path not found: {partial_path}")
                return 1
            result = await runner.run_eval_phase(partial_path, sample)
        else:
            print(f"Error: unknown --phase {phase!r}")
            return 1
    except InfraFailure as e:
        print()
        print("=" * 70)
        print(f"InfraFailure: {e}")
        print(f"No log file written; dispatcher will re-attempt on next scan.")
        print("=" * 70)
        return 3

    print()
    print("=" * 70)
    print(f"Score: {result.score}")
    print(f"Passed: {result.passed}")
    print(f"Duration: {result.total_duration_seconds:.1f}s")
    if result.agent_error:
        print(f"Error: {result.agent_error}")
    print("=" * 70)

    if phase in ("agent", "eval"):
        return 0
    return 0 if result.passed else 1


async def cmd_experiment(args):
    from eval_pipeline.pipeline.experiment import ExperimentRunner

    runner = ExperimentRunner(
        config_path=args.config_file,
        log_dir=args.log_dir,
        max_parallel=args.max_parallel,
        ordering=args.ordering,
        skip_teardown=args.skip_teardown,
    )
    results = await runner.run()
    return 0 if results else 1


def main():
    parser = argparse.ArgumentParser(description="Evaluation pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a single episode")
    run_parser.add_argument("--agent", required=True, choices=AVAILABLE_AGENTS)
    run_parser.add_argument("--benchmark", required=True, choices=AVAILABLE_BENCHMARKS)
    run_parser.add_argument("--task-id", required=True)
    run_parser.add_argument("--benchmark-split", default=None,
                            help="Override benchmark split (swe_bench_live: "
                                 "lite|verified|test|full; default lite).")
    run_parser.add_argument("--model", help="Override LLM model")
    run_parser.add_argument("--base-url", help="vLLM endpoint URL")
    run_parser.add_argument("--api-key", default="dummy", help="API key (default: dummy)")
    run_parser.add_argument("--agent-mode", default="", help="Agent mode (free-form runner hint)")
    run_parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    run_parser.add_argument("--run-timeout", type=int)
    run_parser.add_argument("--eval-timeout", type=int, default=None,
                            help="Wall-cap for a single benchmark.evaluate() "
                                 "call (default: swe_bench=1800s).")
    run_parser.add_argument("--eval-max-retries", type=int, default=None,
                            help="Extra attempts on transient eval-side "
                                 "failure (harness timeout, DNS, no-report). "
                                 "Default: 2 (= 3 attempts total).")
    run_parser.add_argument("--skip-teardown", action="store_true")
    run_parser.add_argument(
        "--resource-limits", default=None,
        action=argparse.BooleanOptionalAction,
        help="Apply per-container resource caps to the agent + eval "
             "containers for host-independent fairness (default: on). "
             "Use --no-resource-limits to run unbounded.")
    run_parser.add_argument("--limit-cpus", type=float, default=None,
                            help="CPU-core ceiling per container (default 4.0)")
    run_parser.add_argument("--limit-memory", default=None,
                            help="Hard memory cap per container (default 16g)")
    run_parser.add_argument("--limit-memory-swap", default=None,
                            help="memory+swap cap; == --limit-memory disables "
                                 "swap for deterministic OOM (default 16g)")
    run_parser.add_argument("--limit-pids", type=int, default=None,
                            help="Max PIDs per container (default 4096)")
    run_parser.add_argument("--limit-shm", default=None,
                            help="/dev/shm size per container (default 2g)")
    run_parser.add_argument("--limit-nofile", default=None,
                            help="ulimit nofile soft:hard (default 65536:65536)")
    run_parser.add_argument("--log-dir", default="logs")
    run_parser.add_argument("--temperature", type=float, default=None,
                            help="Sampling temperature (default 0.6)")
    run_parser.add_argument("--top-p", type=float, default=None,
                            help="Sampling top_p (default 0.95)")
    run_parser.add_argument("--top-k", type=int, default=None,
                            help="Sampling top_k (default 20, vLLM extra_body)")
    run_parser.add_argument("--min-p", type=float, default=None,
                            help="Sampling min_p (default 0.0, vLLM extra_body)")
    run_parser.add_argument("--presence-penalty", type=float, default=None,
                            help="Presence penalty (default 0.0)")
    run_parser.add_argument("--repetition-penalty", type=float, default=None,
                            help="Repetition penalty (default 1.0, vLLM extra_body)")
    run_parser.add_argument("--max-observation-length", type=int, default=None,
                            help="Per-tool-output truncation cap, chars "
                                 "(default 16000; BOAD config = 30000)")
    run_parser.add_argument("--max-iterations", type=int, default=None,
                            help="Max agent turns (default varies by runner; "
                                 "BOAD per_instance_call_limit = 60)")
    run_parser.add_argument("--bash-timeout", type=int, default=None,
                            help="Per-bash-call wallclock cap, seconds "
                                 "(default 1800 = 30min; per-runner config)")
    run_parser.add_argument("--stuck-rollback-turns", type=int, default=None,
                            help="If set, on stuck-detection roll back N "
                                 "turns and retry instead of terminate "
                                 "(vanilla only).")
    run_parser.add_argument("--max-rollbacks", type=int, default=None,
                            help="Max consecutive rollbacks before falling "
                                 "back to terminate (default 3).")
    run_parser.add_argument(
        "--memory-strategy", default=None,
        choices=["branching", "keep_all"],
        help=(
            "Memory policy for the librarian sub-agent. "
            "Default = 'branching' (existing DROP-on-low-new-chars). "
            "Other choices are ablation baselines."
        ),
    )
    run_parser.add_argument(
        "--drop-below-threshold", default=None,
        action=argparse.BooleanOptionalAction,
        help=("BranchThresholdStrategy drop of below-threshold librarian "
              "wakes (default True). Disable with "
              "--no-drop-below-threshold → forces memory_strategy='keep_all'."),
    )
    run_parser.add_argument(
        "--retrieval-backend", default=None,
        choices=["bm25", "sfr_code"],
        help=("Retrieval backend for librarian_rag's RAGNavigator. "
              "Default = 'bm25'. 'sfr_code' requires --retrieval-endpoint."),
    )
    run_parser.add_argument(
        "--retrieval-threshold", type=float, default=None,
        help=("Minimum relevance score for a corpus hit to be prepended "
              "(librarian_rag only). Default = 0.0 (any non-zero overlap)."),
    )
    run_parser.add_argument(
        "--retrieval-max-pairs", type=int, default=None,
        help=("Top-k cap on (command, observation) pairs prepended per "
              "librarian_rag wake. Default = 5."),
    )
    run_parser.add_argument(
        "--retrieval-endpoint", default=None,
        help=("HTTP endpoint for dense retrieval backends "
              "(e.g. http://localhost:19001 for sfr_code). Ignored by "
              "the bm25 backend."),
    )
    run_parser.add_argument("--docker-host", default=None,
                            help="Override DOCKER_HOST for every docker "
                                 "subprocess in this episode "
                                 "(e.g. ssh://my-remote-host). Default = local.")
    run_parser.add_argument("--phase", default="both",
                            choices=["both", "agent", "eval"],
                            help="Episode phase (default: both = "
                                 "legacy single-shot). 'agent' writes "
                                 "<basename>.partial.json then exits; "
                                 "'eval' needs --partial-path.")
    run_parser.add_argument("--partial-path", default=None,
                            help="Path to a *.partial.json from a prior "
                                 "--phase agent run. Required with "
                                 "--phase eval.")

    exp_parser = subparsers.add_parser("experiment", help="Run batch experiment")
    exp_parser.add_argument("config_file")
    exp_parser.add_argument("--ordering", default="sequential", choices=["sequential", "random"])
    exp_parser.add_argument("--max-parallel", type=int, default=4)
    exp_parser.add_argument("--skip-teardown", action="store_true")
    exp_parser.add_argument("--log-dir", default="logs")

    args = parser.parse_args()

    if args.command == "run":
        return asyncio.run(cmd_run(args))
    elif args.command == "experiment":
        return asyncio.run(cmd_experiment(args))


if __name__ == "__main__":
    sys.exit(main())
