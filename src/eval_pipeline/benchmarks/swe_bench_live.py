"""SWE-bench Live evaluator — delegates scoring to the vendored Microsoft harness.

Differences from upstream SWE-bench (see swe_bench.py):

1. Dataset:          SWE-bench-Live/SWE-bench-Live, split = "lite" (BOAD parity).
2. Docker images:    starryzhang/sweb.eval.x86_64.<lowered_id>:latest, where
                     lowered_id = instance_id.replace("__","_1776_").lower().
                     Note the .lower() — upstream keeps original case.
3. Predictions:      a single JSON object keyed by instance_id, value
                     {"model_patch": <diff>}. NOT JSONL; no model_name_or_path.
4. Harness:          microsoft/SWE-bench-Live + microsoft/RepoLaunch, vendored
                     under third_party/sweb_live_vendored/. Invoked as
                     ``python -m evaluation.evaluation`` with cwd at the
                     vendored repo root (the harness does
                     sys.path.insert(0, os.path.join(os.getcwd(), "launch")) at
                     import time, so a different cwd breaks imports).
5. Report:           per-instance file at <output_dir>/<instance_id>/report.json
                     with {"resolved": bool, ...}.
"""

from __future__ import annotations

import asyncio
import json
import platform as _pyplatform
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from eval_pipeline.benchmarks.base import (
    BenchmarkEvaluator, EvalConfig, EvalResult, InfraFailure, Sample,
)


# Path to the vendored harness (third_party/sweb_live_vendored/).
_VENDORED_HARNESS_ROOT = (
    Path(__file__).resolve().parents[3] / "third_party" / "sweb_live_vendored"
)


def _get_arch() -> str:
    # SWE-bench Live publishes x86_64 (Linux) and "win" (Windows) only — no arm64.
    return "x86_64"


def _get_image_name(instance_id: str) -> str:
    """starryzhang/sweb.eval.x86_64.<lowered_id>:latest.

    The .lower() and ``__`` → ``_1776_`` substitution match the
    ``get_default_image_name`` function in the upstream harness
    (evaluation/evaluation.py).
    """
    arch = _get_arch()
    safe_id = instance_id.replace("__", "_1776_").lower()
    return f"starryzhang/sweb.eval.{arch}.{safe_id}:latest"


class SWEBenchLiveEvaluator(BenchmarkEvaluator):
    """SWE-bench Live: continuously-updated GitHub issue resolution benchmark.

    Defaults to the ``lite`` split (300 instances) — matches the BOAD paper
    (arXiv:2512.23631 §5.1: "All results in Tables 2 and 3 are reported on
    Verified and Live (lite) splits").
    """

    def __init__(
        self,
        dataset: str = "SWE-bench-Live/SWE-bench-Live",
        split: str = "lite",
        harness_root: str | Path | None = None,
    ):
        self.dataset = dataset
        self.split = split
        self.harness_root = Path(harness_root) if harness_root else _VENDORED_HARNESS_ROOT
        self._containers: dict[str, str] = {}  # sample_id -> container_name
        # Live harness's internal TIMEOUT is 90min/instance, so this needs
        # to be ≥ that. 6000s gives a small margin over the 5400s upstream
        # cap.
        self.eval_config = EvalConfig(timeout=6000)

    def get_samples(self, limit: int | None = None) -> list[Sample]:
        from datasets import load_dataset

        ds = load_dataset(self.dataset, split=self.split)
        samples = []
        for item in ds:
            # Mirror upstream RepoLaunch's test_cmd hint verbatim (see
            # third_party/sweb_live_vendored/launch/agent/setup/verify.py:153).
            test_cmds = item.get("test_cmds") or ""
            test_cmd_hints = (
                f"\n\nHint: This is the test commands used for this repo other "
                f"developers used in other platforms that may help you find and "
                f"run test cases. <test>{test_cmds}</test>"
                if test_cmds else ""
            )
            instruction = (
                f"Please fix the following issue in the repository at /testbed.\n\n"
                f"<issue>\n{item['problem_statement']}\n</issue>\n\n"
                f"When you're done, make sure all your changes are committed with `git add -A && git commit`."
                f"{test_cmd_hints}"
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
                    # SWE-bench Live extras (not present in upstream Verified)
                    "test_cmds": item.get("test_cmds"),
                    "log_parser": item.get("log_parser"),
                    "created_at": item.get("created_at"),
                    "pull_number": item.get("pull_number"),
                },
            ))
        if limit:
            samples = samples[:limit]
        return samples

    async def setup_container(self, sample: Sample) -> tuple[str, str]:
        image = _get_image_name(sample.id)
        uid = uuid.uuid4().hex[:8]
        container = f"swebenchlive-{sample.id.replace('__', '-').lower()}-{uid}"

        env = self._docker_env()
        await _docker("rm", "-f", container, env=env)

        proc = await asyncio.create_subprocess_exec(
            "docker", "run", "-d", "--name", container,
            "--network", "host",
            "-w", "/testbed",
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
        """Build the env dict for docker subprocess calls.

        Mirrors :meth:`DockerExecutor._subprocess_env`: returns ``None``
        when ``self.docker_host`` is not set so the child inherits the
        parent env unchanged. With it set, returns a copy of os.environ
        with ``DOCKER_HOST`` overridden so the daemon picked by the
        routing layer is used.
        """
        import os
        if not self.docker_host:
            return None
        env = os.environ.copy()
        env["DOCKER_HOST"] = self.docker_host
        return env

    async def evaluate(self, sample: Sample, patch: str | None) -> EvalResult:
        if not patch:
            return EvalResult(score=0.0, passed=False, raw_output="No patch generated")

        if not (self.harness_root / "evaluation" / "evaluation.py").exists():
            return EvalResult(
                score=0.0, passed=False,
                raw_output=(
                    f"SWE-bench Live harness not found at {self.harness_root}. "
                    f"Expected vendored copy under third_party/sweb_live_vendored/."
                ),
            )

        with tempfile.TemporaryDirectory(prefix="swebenchlive_eval_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Predictions file: single JSON dict keyed by instance_id.
            pred_path = tmpdir_path / "predictions.json"
            with open(pred_path, "w") as f:
                json.dump({sample.id: {"model_patch": patch}}, f)

            output_dir = tmpdir_path / "out"
            output_dir.mkdir()

            # Run via the adapter driver, not `-m evaluation.evaluation`:
            # upstream apply_patch bind-mounts a local dir into the eval
            # container, which silently no-ops the patch when DOCKER_HOST
            # is a remote daemon. The driver swaps it for put_archive.
            driver = Path(__file__).resolve().parent / "_swebl_eval_driver.py"
            cmd = [
                sys.executable, str(driver),
                "--dataset", self.dataset,
                "--split", self.split,
                "--platform", "linux",
                "--patch_dir", str(pred_path),
                "--output_dir", str(output_dir),
                "--workers", "1",
                "--overwrite", "1",
                "--instance_ids", sample.id,
            ]

            print(f"  Running SWE-bench Live harness for {sample.id}...")
            # Forward DOCKER_HOST so the harness's docker SDK picks it up.
            harness_env = None
            if self.docker_host:
                import os as _os
                harness_env = _os.environ.copy()
                harness_env["DOCKER_HOST"] = self.docker_host
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.harness_root),  # required: harness uses os.getcwd()
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

            instance_out = output_dir / sample.id
            report_path = instance_out / "report.json"
            if not report_path.exists():
                return EvalResult(
                    score=0.0, passed=False,
                    raw_output=(
                        f"Report not found at {report_path}. "
                        f"stdout: {stdout.decode()[-2000:]}"
                    ),
                )

            with open(report_path) as f:
                report = json.load(f)
            resolved = bool(report.get("resolved", False))

            # The harness produces three artifacts per instance:
            #   - report.json         summary (resolved + PASS_TO_PASS / FAIL_TO_PASS lists)
            #   - status.json         per-test status mapping
            #   - post_patch_log.txt  full pytest stdout from the run (often
            #                         the most informative for debugging:
            #                         which assertion failed, traceback, etc).
            # SWE-bench Verified's harness emits the equivalent pytest
            # output in its raw stdout, which we save verbatim. To match
            # that, we fold the post-patch test log into raw_output here
            # and keep ``report.json`` as eval_metadata.
            log_text = ""
            log_path = instance_out / "post_patch_log.txt"
            if log_path.exists():
                try:
                    log_text = log_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    log_text = f"(failed to read post_patch_log.txt: {e})"
            status_obj: dict | None = None
            status_path = instance_out / "status.json"
            if status_path.exists():
                try:
                    status_obj = json.loads(status_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            raw_parts = []
            raw_parts.append("=== report.json ===")
            raw_parts.append(json.dumps(report, indent=2))
            if status_obj is not None:
                raw_parts.append("\n=== status.json ===")
                raw_parts.append(json.dumps(status_obj, indent=2))
            if log_text:
                raw_parts.append("\n=== post_patch_log.txt ===")
                raw_parts.append(log_text)
            raw_parts.append("\n=== harness stdout (tail) ===")
            raw_parts.append(stdout.decode("utf-8", errors="replace")[-4000:])

            # Normalize the bucket layout so log_viewer's renderer (which
            # reads ``eval_metadata.tests_status.{FAIL,PASS}_TO_{PASS,FAIL}``,
            # the SWE-bench Verified shape) finds them. Upstream Live only
            # reports F2P + P2P; F2F and P2F are absent (no regression
            # bucket in the harness output), so we leave those out and the
            # viewer renders them as 0/0 — that's accurate, not missing.
            tests_status = {}
            for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                bucket = report.get(key)
                if isinstance(bucket, dict):
                    tests_status[key] = {
                        "success": list(bucket.get("success") or []),
                        "failure": list(bucket.get("failure") or []),
                    }
            metadata = dict(report)
            metadata["tests_status"] = tests_status
            if status_obj is not None:
                metadata["status"] = status_obj

            return EvalResult(
                score=1.0 if resolved else 0.0,
                passed=resolved,
                raw_output="\n".join(raw_parts),
                metadata=metadata,
            )

    async def teardown(self, sample: Sample) -> None:
        container = self._containers.pop(sample.id, None)
        if container:
            await _docker("rm", "-f", container, env=self._docker_env())


async def _docker(*args: str, env: dict[str, str] | None = None) -> None:
    proc = await asyncio.create_subprocess_exec(
        "docker", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    await proc.communicate()
