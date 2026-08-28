from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from eval_pipeline.agents import create_agent
from eval_pipeline.benchmarks import create_benchmark
from eval_pipeline.extractors import default_extractors
from eval_pipeline.pipeline.episode import EpisodeRunner
from eval_pipeline.pipeline.schema import EpisodeData


class ExperimentRunner:

    def __init__(
        self,
        config_path: str | Path,
        log_dir: str | Path = "logs",
        max_parallel: int = 4,
        ordering: str = "sequential",
        skip_teardown: bool = False,
    ):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.log_dir = Path(log_dir)
        self.max_parallel = max_parallel
        self.ordering = ordering
        self.skip_teardown = skip_teardown

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_title = self.config.get("title", "exp").replace(" ", "_").lower()[:30]
        self.experiment_dir = self.log_dir / f"exp_{safe_title}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> list[EpisodeData]:
        title = self.config.get("title", "Unnamed")
        print("=" * 70)
        print(f"Experiment: {title}")
        print(f"Log Dir: {self.experiment_dir}")
        print("=" * 70)
        print()

        episodes = []
        for run_spec in self.config.get("runs", []):
            agent_cfg = run_spec["agent"]
            agent_name = agent_cfg.pop("name")
            benchmark_name = run_spec.get("benchmark", "swe_bench")
            benchmark_cfg = run_spec.get("benchmark_config", {})

            agent = create_agent(agent_name, **agent_cfg)
            benchmark = create_benchmark(benchmark_name, **benchmark_cfg)
            samples = benchmark.get_samples()

            task_ids = run_spec.get("task_ids", [])
            if task_ids:
                samples = [s for s in samples if s.id in task_ids]

            for sample in samples:
                episodes.append((agent, benchmark, sample))

        if self.ordering == "random":
            random.shuffle(episodes)

        print(f"Total episodes: {len(episodes)}")
        print()

        semaphore = asyncio.Semaphore(self.max_parallel)
        results: list[EpisodeData] = []
        extractors = default_extractors()

        async def run_one(agent, benchmark, sample, idx):
            async with semaphore:
                print(f"[{idx+1}/{len(episodes)}] {type(agent).__name__} on {sample.id}")
                runner = EpisodeRunner(
                    agent=agent,
                    benchmark=benchmark,
                    extractors=extractors,
                    log_dir=self.experiment_dir,
                    skip_teardown=self.skip_teardown,
                )
                return await runner.run(sample)

        tasks = [
            run_one(a, b, s, i) for i, (a, b, s) in enumerate(episodes)
        ]
        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        for output in outputs:
            if isinstance(output, EpisodeData):
                results.append(output)
            elif isinstance(output, Exception):
                print(f"Episode error: {output}")

        self._save_summary(results)
        self._print_summary(results, len(episodes))

        return results

    def _save_summary(self, results: list[EpisodeData]) -> None:
        summary = {
            "title": self.config.get("title", ""),
            "description": self.config.get("description", ""),
            "total_episodes": len(results),
            "passed": sum(1 for r in results if r.passed),
            "mean_score": sum(r.score for r in results) / len(results) if results else 0.0,
            "episodes": [
                {
                    "task_id": r.task_id,
                    "agent": r.agent,
                    "score": r.score,
                    "passed": r.passed,
                    "duration_seconds": r.total_duration_seconds,
                    "error": r.agent_error,
                }
                for r in results
            ],
        }
        path = self.experiment_dir / "_experiment_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary saved: {path}")

    def _print_summary(self, results: list[EpisodeData], total: int) -> None:
        print()
        print("=" * 70)
        print(f"Completed: {len(results)}/{total}")
        if results:
            passed = sum(1 for r in results if r.passed)
            scores = [r.score for r in results]
            print(f"Passed: {passed}/{len(results)}")
            print(f"Mean Score: {sum(scores) / len(scores):.3f}")
        print("=" * 70)
