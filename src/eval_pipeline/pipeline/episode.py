from __future__ import annotations

import asyncio
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from eval_pipeline.agents.base import AgentResult, AgentRunner
from eval_pipeline.agents.execution import StandardExecution
from eval_pipeline.agents.runtime import (
    ContextWindowExceeded,
    EpisodeStuckAbort,
)
from eval_pipeline.benchmarks.base import (
    BenchmarkEvaluator, EvalResult, InfraFailure, Sample,
)
from eval_pipeline.extractors.base import Extractor
from eval_pipeline.pipeline.schema import EpisodeData
from eval_pipeline.utils.docker import (
    apply_patch_to_container,
    DockerExecutor,
)
from eval_pipeline.utils.routing import get_router


def _merge_live_http(llm_calls: list[dict], http_path: Path) -> tuple[bool, bool]:
    """Fold the raw HTTP capture (``live_http.jsonl``) into each turn dict.

    The sidecar is written by the LLM client wrapper (the only layer that
    sees the bytes vLLM exchanged), one JSON line per call. Here — at the
    pipeline layer, which owns the episode file — we attach each call's
    ``raw_prompt`` (the rendered prompt string the model actually received)
    and ``raw_completion`` (vLLM's output stitched verbatim, no client
    re-parse) and ``reasoning`` (the chain-of-thought vLLM's reasoning parser
    split out of ``content``; None when thinking is off or no parser is
    active) onto the matching turn, then drop the now-redundant parsed
    ``llm_input``: ``raw_prompt`` is its raw equivalent and carries the same
    (full, per-turn) history, so keeping both would double the input bulk.

    Matching is by ``response_id`` (turn ``raw_output[].provider_data`` ↔
    sidecar ``response.id``) — robust to MAS sub-agents interleaving into one
    shared sidecar. No positional fallback: an unmatched turn is left
    untouched (keeps its ``llm_input``) rather than risk misattribution.

    Idempotent: a turn that already carries ``raw_prompt`` is skipped, so
    re-saving an already-folded episode is a no-op.

    Returns ``(enriched, complete)``. ``enriched`` is True if any turn was
    folded. ``complete`` is True only if *every* turn that names a
    ``response_id`` ended up folded — the caller deletes the sidecar only
    then, so a partial match never strands an unmatched turn's raw bytes.
    (Turns with no ``response_id`` — failed/empty calls — carry no raw and
    don't block deletion.)
    """
    if not llm_calls or not http_path.exists():
        return False, False
    recs: list[dict] = []
    for line in http_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except Exception:
            pass
    if not recs:
        return False, False

    by_id: dict[str, dict] = {}
    for r in recs:
        rid = ((r.get("response") or {}).get("id")) if isinstance(r, dict) else None
        if rid:
            by_id.setdefault(rid, r)

    def _turn_response_id(call: dict) -> str | None:
        for item in call.get("raw_output") or []:
            pd = (item or {}).get("provider_data") or {}
            if pd.get("response_id"):
                return pd["response_id"]
        return None

    enriched = False
    unmatched_with_id = 0
    for call in llm_calls:
        if call.get("raw_prompt") is not None or call.get("raw_completion") is not None:
            continue  # already folded
        rid = _turn_response_id(call)
        rec = by_id.get(rid) if rid else None
        if rec is None:
            if rid is not None:
                unmatched_with_id += 1  # had an id but no sidecar record
            continue
        call["raw_prompt"] = rec.get("raw_prompt")
        call["raw_completion"] = rec.get("raw_completion")
        call["reasoning"] = rec.get("reasoning")
        call.pop("llm_input", None)  # raw_prompt supersedes the parsed history
        enriched = True
    return enriched, (enriched and unmatched_with_id == 0)


class EpisodeRunner:

    def __init__(
        self,
        agent: AgentRunner,
        benchmark: BenchmarkEvaluator,
        extractors: list[Extractor] | None = None,
        log_dir: str | Path = "logs",
        skip_teardown: bool = False,
        docker_host: str | None = None,
    ):
        self.agent = agent
        self.benchmark = benchmark
        self.extractors = extractors or []
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.skip_teardown = skip_teardown
        self.docker_host = docker_host


    async def run_agent_phase(self, sample: Sample) -> tuple[EpisodeData, str | None]:
        start_time = datetime.utcnow()
        container_name = None
        work_dir = ""
        agent_result = AgentResult()

        try:
            docker_host = self.docker_host
            if docker_host is None:
                try:
                    router = get_router()
                    docker_host = router.host_for(sample.id) if router.enabled else None
                except FileNotFoundError:
                    docker_host = None
            if docker_host:
                print(f"Routing to {docker_host}")
            self.agent.docker_host = docker_host
            self.benchmark.docker_host = docker_host

            print("Setting up container...")
            try:
                container_name, work_dir = await self.benchmark.setup_container(sample)
            except InfraFailure as e:
                print(f"InfraFailure during setup: {e}")
                try:
                    await self.benchmark.teardown(sample)
                except Exception:
                    pass
                raise

            print(f"Installing agent in {container_name}...")
            try:
                install_ok = await self.agent.setup(container_name, work_dir)
                if not install_ok:
                    raise RuntimeError("Agent installation failed")

                print("Running agent...", flush=True)
                try:
                    self.agent._live_log_dir = str(self.log_dir)
                    docker = getattr(self.agent, "docker", None)
                    if docker is not None:
                        try:
                            docker.live_bash_log_path = str(
                                self.log_dir / "live_bash.jsonl"
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
                agent_result = await self._invoke_agent(
                    container_name, sample.instruction, work_dir,
                )

                if not agent_result.patch:
                    agent_result.patch = await self._extract_patch(
                        container_name, work_dir,
                    )

                if hasattr(self.agent, "_swebench_image"):
                    await self._ensure_container_running(container_name)
                    if agent_result.patch:
                        applied = await apply_patch_to_container(
                            container_name, agent_result.patch, work_dir,
                            docker_host=docker_host,
                        )
                        if not applied:
                            print("  Warning: patch apply to eval container failed")
            except Exception as e:
                if not agent_result.error:
                    agent_result.error = f"{type(e).__name__}: {e}"
                print(f"Agent-phase error: {agent_result.error}")

        except InfraFailure:
            raise

        if container_name:
            try:
                await self.benchmark.teardown(sample)
                container_name = None
            except Exception as e:
                print(f"Phase-A teardown warning: {e}")

        episode = EpisodeData(
            task_id=sample.id,
            benchmark=getattr(self.benchmark, "task_type", "") or type(self.benchmark).__name__,
            agent=type(self.agent).__name__,
            model=getattr(self.agent, "model", ""),
            agent_mode=getattr(self.agent, "agent_mode", ""),
            seed=getattr(self.agent, "_seed", 0),
            timestamp=start_time.isoformat(),
            patch=agent_result.patch,
            agent_duration_seconds=agent_result.duration_seconds,
            agent_error=agent_result.error,
            score=0.0,
            passed=False,
            eval_raw_output="",
            eval_metadata={
                "_phase_a": {
                    "container_name": container_name,
                    "work_dir": work_dir,
                    "docker_host": docker_host,
                    "started_at": start_time.isoformat(),
                },
            },
            total_duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            resource_limits=(
                self.benchmark.resource_limits.as_dict()
                if getattr(self.benchmark, "resource_limits", None) else None
            ),
            agent_raw_output=agent_result.raw_output,
            llm_calls=agent_result.llm_calls or None,
        )

        partial_path = self._partial_path(episode, sample)
        # The agent has finished, so the sidecar is complete: fold raw HTTP
        # into the partial now (it becomes self-contained; reeval re-saves
        # the already-folded copy).
        self._fold_and_save(episode, partial_path)
        print(f"Saved partial: {partial_path}")

        return episode, container_name

    async def run_eval_phase(
        self,
        partial_path: Path,
        sample: Sample,
    ) -> EpisodeData:
        partial_path = Path(partial_path)
        partial = EpisodeData.load(partial_path)

        meta = partial.eval_metadata or {}
        phase_a = meta.get("_phase_a", {}) or {}
        container_name = phase_a.get("container_name")
        docker_host = self.docker_host or phase_a.get("docker_host")
        if self.docker_host and self.docker_host != phase_a.get("docker_host"):
            container_name = None

        if container_name:
            try:
                self.benchmark._containers[sample.id] = container_name  # type: ignore[attr-defined]
            except Exception:
                pass
        if docker_host:
            self.benchmark.docker_host = docker_host
            try:
                self.agent.docker_host = docker_host
            except Exception:
                pass

        eval_result = EvalResult(score=0.0, passed=False)
        try:
            if container_name:
                await self._ensure_container_running(container_name)
            print(
                f"Evaluating (timeout={self.benchmark.eval_config.timeout}s, "
                f"retries={self.benchmark.eval_config.max_retries})...",
                flush=True,
            )
            eval_result = await self.benchmark.evaluate_with_retry(
                sample, partial.patch,
            )
        except InfraFailure:
            try:
                if not self.skip_teardown:
                    await self.benchmark.teardown(sample)
            except Exception:
                pass
            raise
        except Exception as e:
            eval_result = EvalResult(
                score=0.0, passed=False,
                raw_output=f"Harness failed (outer): {type(e).__name__}: {e}",
            )
            print(f"Eval-phase outer error: {eval_result.raw_output}")
        finally:
            if not self.skip_teardown:
                try:
                    await self.benchmark.teardown(sample)
                except Exception as e:
                    print(f"Teardown warning: {e}")
            elif container_name:
                print(f"Skipping teardown: {container_name}")

        partial.score = eval_result.score
        partial.passed = eval_result.passed
        partial.eval_raw_output = eval_result.raw_output
        partial.eval_metadata = dict(eval_result.metadata or {})

        agent_result_proxy = AgentResult(
            patch=partial.patch,
            raw_output=partial.agent_raw_output,
            exit_code=0,
            duration_seconds=partial.agent_duration_seconds,
            llm_calls=partial.llm_calls or [],
            error=partial.agent_error,
        )
        for extractor in self.extractors:
            try:
                extractor.enrich(partial, agent_result_proxy, eval_result)
            except Exception as e:
                print(f"  Extractor {type(extractor).__name__} failed: {e}")

        final_path = self._final_path_from_partial(partial_path)
        # Partial was already folded at phase-A save; this is a no-op fold
        # (idempotent) that just writes the final file.
        self._fold_and_save(partial, final_path)
        try:
            partial_path.unlink()
        except FileNotFoundError:
            pass
        print(f"Saved: {final_path}")
        return partial

    def _partial_path(self, episode: EpisodeData, sample: Sample) -> Path:
        safe_id = sample.id.replace("/", "_").replace("\\", "_")
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        mode_str = f"_{episode.agent_mode}" if episode.agent_mode else ""
        seed_str = f"_s{episode.seed}" if episode.seed else ""
        filename = (
            f"{episode.benchmark}_{safe_id}_{episode.agent}{mode_str}{seed_str}_{ts}"
            ".partial.json"
        )
        return self.log_dir / filename

    @staticmethod
    def _final_path_from_partial(partial: Path) -> Path:
        name = partial.name
        if name.endswith(".partial.json"):
            name = name[: -len(".partial.json")] + ".json"
        return partial.with_name(name)


    async def run(self, sample: Sample) -> EpisodeData:
        start_time = datetime.utcnow()
        container_name = None
        agent_result = AgentResult()
        eval_result = EvalResult(score=0.0, passed=False)

        try:
            docker_host = self.docker_host
            if docker_host is None:
                try:
                    router = get_router()
                    docker_host = router.host_for(sample.id) if router.enabled else None
                except FileNotFoundError:
                    docker_host = None
            if docker_host:
                print(f"Routing to {docker_host}")
            self.agent.docker_host = docker_host
            self.benchmark.docker_host = docker_host

            print(f"Setting up container...")
            try:
                container_name, work_dir = await self.benchmark.setup_container(sample)
            except InfraFailure as e:
                print(f"InfraFailure during setup: {e}")
                print(f"Skipping log persistence so dispatcher can retry.")
                if not self.skip_teardown:
                    try:
                        await self.benchmark.teardown(sample)
                    except Exception:
                        pass
                raise

            print(f"Installing agent in {container_name}...")
            try:
                install_ok = await self.agent.setup(container_name, work_dir)
                if not install_ok:
                    raise RuntimeError("Agent installation failed")

                print(f"Running agent...", flush=True)
                try:
                    self.agent._live_log_dir = str(self.log_dir)
                    docker = getattr(self.agent, "docker", None)
                    if docker is not None:
                        try:
                            docker.live_bash_log_path = str(
                                self.log_dir / "live_bash.jsonl"
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
                agent_result = await self._invoke_agent(
                    container_name, sample.instruction, work_dir,
                )

                if not agent_result.patch:
                    agent_result.patch = await self._extract_patch(
                        container_name, work_dir,
                    )

                if hasattr(self.agent, '_swebench_image'):
                    await self._ensure_container_running(container_name)
                    if agent_result.patch:
                        applied = await apply_patch_to_container(
                            container_name, agent_result.patch, work_dir,
                            docker_host=docker_host,
                        )
                        if not applied:
                            print("  Warning: patch apply to eval container failed")
            except Exception as e:
                if not agent_result.error:
                    agent_result.error = f"{type(e).__name__}: {e}"
                print(f"Agent-phase error: {agent_result.error}")

            try:
                await self._ensure_container_running(container_name)
                print(f"Evaluating (timeout={self.benchmark.eval_config.timeout}s, "
                      f"retries={self.benchmark.eval_config.max_retries})...",
                      flush=True)
                eval_result = await self.benchmark.evaluate_with_retry(
                    sample, agent_result.patch,
                )
            except InfraFailure:
                raise
            except Exception as e:
                eval_result = EvalResult(
                    score=0.0, passed=False,
                    raw_output=f"Harness failed (outer): "
                               f"{type(e).__name__}: {e}",
                )
                print(f"Eval-phase outer error: {eval_result.raw_output}")

        finally:
            if not self.skip_teardown:
                try:
                    await self.benchmark.teardown(sample)
                except Exception as e:
                    print(f"Teardown warning: {e}")
            elif container_name:
                print(f"Skipping teardown: {container_name}")

        end_time = datetime.utcnow()

        episode = EpisodeData(
            task_id=sample.id,
            benchmark=getattr(self.benchmark, 'task_type', '') or type(self.benchmark).__name__,
            agent=type(self.agent).__name__,
            model=getattr(self.agent, 'model', ''),
            agent_mode=getattr(self.agent, 'agent_mode', ''),
            seed=getattr(self.agent, '_seed', 0),
            timestamp=start_time.isoformat(),
            patch=agent_result.patch,
            agent_duration_seconds=agent_result.duration_seconds,
            agent_error=agent_result.error,
            score=eval_result.score,
            passed=eval_result.passed,
            eval_raw_output=eval_result.raw_output,
            eval_metadata=eval_result.metadata,
            total_duration_seconds=(end_time - start_time).total_seconds(),
            resource_limits=(
                self.benchmark.resource_limits.as_dict()
                if getattr(self.benchmark, "resource_limits", None) else None
            ),
            agent_raw_output=agent_result.raw_output,
            llm_calls=agent_result.llm_calls or None,
        )

        for extractor in self.extractors:
            try:
                extractor.enrich(episode, agent_result, eval_result)
            except Exception as e:
                print(f"  Extractor {type(extractor).__name__} failed: {e}")

        self._save(episode, sample)

        return episode

    async def _invoke_agent(
        self, container_name: str, instruction: str, work_dir: str,
    ) -> AgentResult:
        runtime = self.agent._make_runtime_config()
        executor = StandardExecution(total_timeout=float(runtime.run_timeout))
        start = time.monotonic()
        error: str | None = None

        try:
            await self.agent.run_with_executor(
                instruction, work_dir,
                executor=executor, runtime=runtime,
            )
        except NotImplementedError:
            return await self.agent.run(container_name, instruction, work_dir)
        except asyncio.TimeoutError:
            error = f"run_timeout: exceeded {runtime.run_timeout}s"
        except ContextWindowExceeded as cwe:
            error = f"context_window_exceeded: {cwe}"
        except EpisodeStuckAbort as e:
            # A wedged sub-agent role hit its consecutive-stuck budget and the
            # delegation escalated to an episode-level abort (BaseException, so
            # the agents-SDK function-tool handler couldn't swallow it). Treat
            # as a graceful terminal like stuck: commit the current patch.
            error = f"stuck_terminate: episode aborted ({e})"
        except Exception as e:
            traceback.print_exc()
            error = f"{type(e).__name__}: {e}"

        patch = (
            await self.agent.docker.get_patch()
            if getattr(self.agent, "docker", None) is not None
            else None
        )
        duration = time.monotonic() - start
        cl = getattr(self.agent, "call_log", None)
        if cl is not None:
            llm_calls = cl.to_dicts()
            totals = cl.total()
        else:
            llm_calls = []
            totals = {}
        # See base.py: llm_calls lives once at the top level only.
        raw_output: dict = {"totals": totals}
        pool = getattr(self.agent, "_librarian_pool", None)
        if pool is not None:
            try:
                raw_output["pool_stats"] = pool.stats()
            except Exception:
                pass
        return AgentResult(
            patch=patch or None,
            raw_output=raw_output,
            exit_code=0,
            duration_seconds=duration,
            llm_calls=llm_calls,
            error=error,
        )

    async def _extract_patch(self, container_name: str, work_dir: str) -> str:
        agent_docker = getattr(self.agent, "_docker", None)
        if agent_docker is not None:
            return await agent_docker.get_patch()
        executor = DockerExecutor(
            container_name, work_dir,
            docker_host=getattr(self.agent, "docker_host", None),
        )
        return await executor.get_patch()

    def _docker_env(self) -> dict[str, str] | None:
        import os
        host = getattr(self.agent, "docker_host", None) or getattr(
            self.benchmark, "docker_host", None,
        )
        if not host:
            return None
        env = os.environ.copy()
        env["DOCKER_HOST"] = host
        return env

    async def _ensure_container_running(self, container_name: str) -> None:
        env = self._docker_env()
        for attempt in range(3):
            proc = await asyncio.create_subprocess_exec(
                "docker", "inspect", "-f", "{{.State.Running}}", container_name,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, _ = await proc.communicate()
            if stdout.decode().strip() == "true":
                return

            proc = await asyncio.create_subprocess_exec(
                "docker", "start", container_name,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            await proc.communicate()
            await asyncio.sleep(3)

    def _fold_and_save(self, episode: EpisodeData, path: Path) -> None:
        """Fold the raw HTTP sidecar into the episode, save, then drop the
        sidecar. Order matters: the file is written *before* the sidecar is
        unlinked, so a save failure never loses the raw capture."""
        http_path = self.log_dir / "live_http.jsonl"
        _enriched, complete = _merge_live_http(episode.llm_calls or [], http_path)
        episode.save(path)
        if complete:
            try:
                http_path.unlink()
            except OSError:
                pass

    def _save(self, episode: EpisodeData, sample: Sample) -> None:
        safe_id = sample.id.replace("/", "_").replace("\\", "_")
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        mode_str = f"_{episode.agent_mode}" if episode.agent_mode else ""
        seed_str = f"_s{episode.seed}" if episode.seed else ""
        filename = f"{episode.benchmark}_{safe_id}_{episode.agent}{mode_str}{seed_str}_{ts}.json"
        path = self.log_dir / filename
        self._fold_and_save(episode, path)
        print(f"Saved: {path}")
