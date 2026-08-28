from __future__ import annotations

import asyncio
import json
import platform
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from eval_pipeline.benchmarks.base import (
    BenchmarkEvaluator, EvalConfig, EvalResult, InfraFailure, Sample,
)
from eval_pipeline.benchmarks.resources import ResourceLimits


def _get_arch() -> str:
    return "arm64" if platform.machine() in {"aarch64", "arm64"} else "x86_64"


def _get_image_name(task_id: str) -> str:
    arch = _get_arch()
    safe_id = task_id.replace("__", "_1776_")
    return f"swebench/sweb.eval.{arch}.{safe_id}:latest"


class SWEBenchEvaluator(BenchmarkEvaluator):

    def __init__(
        self,
        dataset: str = "princeton-nlp/SWE-bench_Verified",
        split: str = "test",
    ):
        self.dataset = dataset
        self.split = split
        self._containers: dict[str, str] = {}
        self.eval_config = EvalConfig(timeout=9000, test_timeout=5400)
        self.resource_limits = ResourceLimits()

    def get_samples(self, limit: int | None = None) -> list[Sample]:
        from datasets import load_dataset

        ds = load_dataset(self.dataset, split=self.split)
        samples = []
        for item in ds:
            instruction = (
                f"Please fix the following issue in the repository at /testbed.\n\n"
                f"<issue>\n{item['problem_statement']}\n</issue>\n\n"
                f"When you're done, make sure all your changes are committed with `git add -A && git commit`."
            )
            samples.append(Sample(
                id=item["instance_id"],
                instruction=instruction,
                metadata={
                    "repo": item.get("repo"),
                    "base_commit": item.get("base_commit"),
                    "patch": item.get("patch"),
                    "test_patch": item.get("test_patch"),
                    "hints_text": item.get("hints_text"),
                    "PASS_TO_PASS": item.get("PASS_TO_PASS"),
                    "FAIL_TO_PASS": item.get("FAIL_TO_PASS"),
                },
            ))
        if limit:
            samples = samples[:limit]
        return samples

    async def setup_container(self, sample: Sample) -> tuple[str, str]:
        image = _get_image_name(sample.id)
        uid = uuid.uuid4().hex[:8]
        container = f"swebench-{sample.id.replace('__', '-')}-{uid}"

        env = self._docker_env()
        await _docker("rm", "-f", container, env=env)

        limit_args = self.resource_limits.docker_run_args()
        print(f"  [swebench] agent container limits: "
              f"{self.resource_limits.describe()}")
        proc = await asyncio.create_subprocess_exec(
            "docker", "run", "-d", "--name", container,
            "--network", "host",
            "-w", "/testbed",
            *limit_args,
            image, "sleep", "infinity",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode()
            if "Unable to find image" in err or "pull access denied" in err \
                    or "manifest unknown" in err:
                raise InfraFailure(
                    f"Image pull failed for {image}: {err.strip()}"
                )
            raise RuntimeError(f"Failed to start container: {err}")

        self._containers[sample.id] = container
        return container, "/testbed"

    def _docker_env(self) -> dict[str, str] | None:
        import os
        if not self.docker_host:
            return None
        env = os.environ.copy()
        env["DOCKER_HOST"] = self.docker_host
        return env

    async def evaluate(self, sample: Sample, patch: str | None) -> EvalResult:
        if not patch:
            return EvalResult(score=0.0, passed=False, raw_output="No patch generated")

        with tempfile.TemporaryDirectory(prefix="swebench_eval_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            pred_path = tmpdir_path / "predictions.jsonl"
            with open(pred_path, "w") as f:
                json.dump({
                    "instance_id": sample.id,
                    "model_patch": patch,
                    "model_name_or_path": "agent",
                }, f)
                f.write("\n")

            run_id = f"eval_{uuid.uuid4().hex[:8]}"
            # Bootstrap: install the docker-SDK resource-limit patch BEFORE the
            # harness creates its eval container, then run the harness module as
            # __main__ (it parses sys.argv and calls main()). The installed
            # swebench sets no container limits, so this is the only hook point.
            bootstrap = (
                "import runpy, sys; "
                "from eval_pipeline.benchmarks._swebench_resource_patch "
                "import install; install(); "
                "sys.argv = ['swebench.harness.run_evaluation'] + sys.argv[1:]; "
                "runpy.run_module('swebench.harness.run_evaluation', "
                "run_name='__main__')"
            )
            cmd = [
                sys.executable, "-c", bootstrap,
                "--predictions_path", str(pred_path),
                "--run_id", run_id,
                "--dataset_name", self.dataset,
                "--split", self.split,
                "--instance_ids", sample.id,
                "--max_workers", "1",
                "--timeout", str(self.eval_config.test_timeout),
            ]

            print(f"  Running swebench harness for {sample.id}...")
            import os as _os
            harness_env = _os.environ.copy()
            # Carry resource limits across the process boundary (read back by
            # _swebench_resource_patch.install via ResourceLimits.from_env).
            harness_env.update(self.resource_limits.to_env())
            # Ensure eval_pipeline is importable from the harness cwd (tmpdir)
            # even if the editable .pth is absent.
            repo_src = str(Path(__file__).resolve().parents[2])
            harness_env["PYTHONPATH"] = (
                repo_src + _os.pathsep + harness_env.get("PYTHONPATH", "")
            )
            if self.docker_host:
                harness_env["DOCKER_HOST"] = self.docker_host
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=tmpdir,
                env=harness_env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.eval_config.timeout,
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                return EvalResult(
                    score=0.0, passed=False,
                    raw_output=f"Harness timed out after "
                               f"{self.eval_config.timeout}s",
                )

            if proc.returncode != 0:
                err = stderr.decode()[-2000:]
                return EvalResult(
                    score=0.0, passed=False,
                    raw_output=f"Harness failed (rc={proc.returncode}): {err}",
                )

            #      test_output.txt — verbatim stdout of the project's
            per_instance = self._find_per_instance_report(
                tmpdir_path, run_id, sample.id,
            )
            test_output = self._read_test_output(
                tmpdir_path, run_id, sample.id,
            )
            summary = self._find_summary_report(tmpdir_path, run_id)

            if per_instance is None and test_output:
                test_log_path = (
                    tmpdir_path / "logs" / "run_evaluation" / run_id
                    / "agent" / sample.id / "test_output.txt"
                )
                if not test_log_path.exists():
                    # Fall back to rglob if upstream layout shifted
                    for p in tmpdir_path.rglob("test_output.txt"):
                        parts = set(p.parts)
                        if run_id in parts and sample.id in parts:
                            test_log_path = p
                            break
                if test_log_path.exists():
                    per_instance = self._compute_report_from_test_output(
                        sample, patch, test_log_path,
                    )
                    if per_instance is not None:
                        print(
                            f"  [recovered] used test_output.txt fallback "
                            f"after harness no-report for {sample.id}"
                        )

            if per_instance is not None:
                resolved = bool(per_instance.get(sample.id, {}).get("resolved"))
            elif summary is not None:
                resolved = sample.id in (
                    summary.get("resolved", [])
                    or summary.get("resolved_ids", [])
                )
            else:
                return EvalResult(
                    score=0.0, passed=False,
                    raw_output=(
                        "No report found (neither per-instance nor "
                        f"summary). stdout: {stdout.decode()[-2000:]}"
                    ),
                )

            metadata: dict = {}
            if per_instance is not None:
                metadata["per_instance_report"] = per_instance
                metadata["tests_status"] = (
                    per_instance.get(sample.id, {}).get("tests_status") or {}
                )
            if summary is not None:
                metadata["summary_report"] = summary
            metadata["resolved"] = resolved
            metadata["instance_id"] = sample.id

            # verbatim test runner stdout (truncated head/tail).
            raw_chunks: list[str] = []
            if per_instance is not None:
                raw_chunks.append("=== per-instance report ===")
                raw_chunks.append(json.dumps(per_instance, indent=2))
            if test_output:
                if len(test_output) > 110_000:
                    head = test_output[:50_000]
                    tail = test_output[-50_000:]
                    test_output_clipped = (
                        head
                        + f"\n\n[... clipped {len(test_output) - 100_000} "
                          "chars from the middle ...]\n\n"
                        + tail
                    )
                else:
                    test_output_clipped = test_output
                raw_chunks.append("=== test runner stdout ===")
                raw_chunks.append(test_output_clipped)
                metadata["test_output_len"] = len(test_output)
            if summary is not None:
                raw_chunks.append("=== summary report ===")
                raw_chunks.append(json.dumps(summary, indent=2))

            return EvalResult(
                score=1.0 if resolved else 0.0,
                passed=resolved,
                raw_output="\n\n".join(raw_chunks),
                metadata=metadata,
            )

    async def teardown(self, sample: Sample) -> None:
        container = self._containers.pop(sample.id, None)
        if container:
            await _docker("rm", "-f", container, env=self._docker_env())

    def _find_per_instance_report(
        self, tmpdir: Path, run_id: str, instance_id: str,
    ) -> dict | None:
        path = (
            tmpdir / "logs" / "run_evaluation" / run_id / "agent"
            / instance_id / "report.json"
        )
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        for p in tmpdir.rglob("report.json"):
            parts = set(p.parts)
            if run_id in parts and instance_id in parts:
                try:
                    with open(p) as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
        return None

    def _read_test_output(
        self, tmpdir: Path, run_id: str, instance_id: str,
    ) -> str | None:
        path = (
            tmpdir / "logs" / "run_evaluation" / run_id / "agent"
            / instance_id / "test_output.txt"
        )
        if path.exists():
            try:
                with open(path) as f:
                    return f.read()
            except OSError:
                pass
        for p in tmpdir.rglob("test_output.txt"):
            parts = set(p.parts)
            if run_id in parts and instance_id in parts:
                try:
                    with open(p) as f:
                        return f.read()
                except OSError:
                    continue
        return None

    def _find_summary_report(self, tmpdir: Path, run_id: str) -> dict | None:
        """Summary report (counts) at ``<cwd>/<model>.<run_id>.json``.

        swebench's ``make_run_report`` writes to
        ``{prediction.model_name_or_path}.{run_id}.json`` — our predictions
        use ``model_name_or_path="agent"`` so the file is
        ``agent.eval_<hex>.json``. The older ``{run_id}.json`` filename is
        also tried as a fallback in case upstream changes again.
        """
        candidates = [
            tmpdir / f"agent.{run_id}.json",
            tmpdir / f"{run_id}.json",
        ]
        for path in candidates:
            if path.exists():
                try:
                    with open(path) as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue
        return None

    def _compute_report_from_test_output(
        self, sample: Sample, patch: str, test_log_path: Path,
    ) -> dict | None:
        try:
            from swebench.harness.grading import get_eval_report
            from swebench.harness.test_spec.test_spec import make_test_spec
        except Exception:
            return None
        meta = sample.metadata or {}
        instance = {
            "instance_id": sample.id,
            "repo": meta.get("repo"),
            "base_commit": meta.get("base_commit"),
            "patch": meta.get("patch"),
            "test_patch": meta.get("test_patch"),
            "FAIL_TO_PASS": meta.get("FAIL_TO_PASS"),
            "PASS_TO_PASS": meta.get("PASS_TO_PASS"),
            "problem_statement": meta.get("problem_statement", ""),
            "hints_text": meta.get("hints_text") or "",
            "version": meta.get("version", ""),
            "environment_setup_commit": meta.get(
                "environment_setup_commit", meta.get("base_commit"),
            ),
            "created_at": meta.get("created_at", ""),
        }
        try:
            test_spec = make_test_spec(instance)
        except Exception:
            return None
        prediction = {
            "instance_id": sample.id,
            "model_name_or_path": "agent",
            "model_patch": patch,
        }
        try:
            report = get_eval_report(
                test_spec=test_spec,
                prediction=prediction,
                test_log_path=str(test_log_path),
                include_tests_status=True,
            )
        except Exception:
            return None
        if isinstance(report, dict) and sample.id in report:
            return report
        return None


async def _docker(*args: str, env: dict[str, str] | None = None) -> None:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    await proc.communicate()
