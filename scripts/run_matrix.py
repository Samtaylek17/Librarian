#!/usr/bin/env python3
"""Round-robin dispatch of a matrix of runs across a pool of vLLM URLs.

Prereqs (see the top-level README "Reproducing the experiments"):
  (i)  A list of live vLLM URLs passed via --urls or --urls-file.
  (ii) A matrix spec YAML — see configs/experiments/README.md for shape.

One worker per URL, sequential inside each worker's queue. Resumable:
runs whose log dir already contains a <Evaluator>_*.json file are skipped.

Usage:
  python scripts/run_matrix.py \
      --urls http://localhost:18001/v1,http://localhost:18002/v1 \
      --spec configs/experiments/my_exp.yaml

  python scripts/run_matrix.py --urls-file urls.txt --spec ...
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = REPO_ROOT / "scripts" / "run.py"
PYTHON = os.environ.get("PYTHON", sys.executable)

sys.path.insert(0, str(REPO_ROOT / "src"))
from eval_pipeline.utils.routing import DispatchRouter  # noqa: E402

ALLOWED_FIELDS = {
    "agent", "benchmark", "benchmark_split", "task_id", "task_type", "model",
    "seed", "run_timeout", "max_iterations", "max_observation_length",
    "bash_timeout",
    "eval_timeout", "eval_max_retries",
    "temperature", "top_p", "top_k", "min_p",
    "presence_penalty", "repetition_penalty",
    "stuck_rollback_turns", "max_rollbacks",
    "branch_max_turns",
    "memory_strategy",
    "limit_cpus", "limit_memory", "limit_memory_swap",
    "drop_below_threshold",
    "retrieval_backend", "retrieval_threshold", "retrieval_max_pairs",
    "retrieval_endpoint",
}

_BOOL_FIELDS = {"drop_below_threshold", "resource_limits"}

_LIBRARIAN_TOGGLE_SHORT = {
    "drop_below_threshold": "dbt",
}


def load_urls(args) -> list[str]:
    if args.urls and args.urls_file:
        sys.exit("Pass exactly one of --urls or --urls-file")
    if args.urls:
        urls = [u.strip() for u in args.urls.split(",") if u.strip()]
    elif args.urls_file:
        urls = [
            ln.strip()
            for ln in Path(args.urls_file).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
    else:
        sys.exit("No vLLM URLs supplied (use --urls or --urls-file)")
    if not urls:
        sys.exit("URL list is empty")
    return urls


def probe_url(url: str, timeout: float = 5.0) -> str | None:
    probe = url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(probe, timeout=timeout) as r:
            import json as _json
            body = _json.load(r)
            data = body.get("data") or []
            if data:
                return data[0].get("id")
    except Exception:
        return None
    return None


def _get_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = json.load(response)
    if not isinstance(body, dict):
        raise ValueError(f"{url} returned a non-object JSON response")
    return body


def probe_server(url: str, timeout: float = 5.0) -> dict:
    """Fetch the model identity and immutable experiment fingerprint."""
    api_url = url.rstrip("/")
    root_url = api_url.removesuffix("/v1")
    models = _get_json(f"{api_url}/models", timeout=timeout)
    info = _get_json(f"{root_url}/experiment-info", timeout=timeout)
    return {
        "url": url,
        "model_ids": [item.get("id") for item in models.get("data") or []],
        "experiment_info": info,
    }


def validate_server_pool(
    servers: list[dict], expected_model: str | None = None,
) -> str:
    if not servers:
        raise ValueError("no server fingerprints were collected")
    errors: list[str] = []
    fingerprints: set[str] = set()
    for server in servers:
        url = server.get("url", "<unknown>")
        model_ids = server.get("model_ids") or []
        info = server.get("experiment_info") or {}
        fingerprint = info.get("fingerprint")
        if not isinstance(fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", fingerprint,
        ):
            errors.append(f"{url}: missing or invalid experiment fingerprint")
        else:
            fingerprints.add(fingerprint)
        if info.get("gpu_count") != 1:
            errors.append(f"{url}: expected one GPU, found {info.get('gpu_count')!r}")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", info.get("image_digest", "")):
            errors.append(f"{url}: image digest is not recorded")
        if not re.fullmatch(
            r"[0-9a-f]{40}", info.get("repository_commit", ""),
        ):
            errors.append(f"{url}: repository commit is not a full commit SHA")
        if info.get("repository_dirty") is not False:
            errors.append(f"{url}: repository working tree is dirty")
        revision = info.get("model_revision")
        if not isinstance(revision, str) or not re.fullmatch(
            r"[0-9a-f]{40}", revision,
        ):
            errors.append(f"{url}: model revision is not a full commit SHA")
        if info.get("tokenizer_revision") != revision:
            errors.append(f"{url}: tokenizer and model revisions differ")
        if expected_model:
            if info.get("model") != expected_model:
                errors.append(
                    f"{url}: metadata model {info.get('model')!r} does not "
                    f"match matrix model {expected_model!r}"
                )
            if expected_model not in model_ids:
                errors.append(
                    f"{url}: /models advertises {model_ids!r}, expected "
                    f"{expected_model!r}"
                )
    if len(fingerprints) > 1:
        errors.append(
            "server compatibility fingerprints differ: "
            + ", ".join(sorted(fingerprints))
        )
    if errors:
        raise ValueError("serving preflight failed:\n  - " + "\n  - ".join(errors))
    return next(iter(fingerprints))


def is_url_healthy(
    url: str,
    timeout: float = 5.0,
    expected_fingerprint: str | None = None,
) -> bool:
    probe = url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(probe, timeout=timeout) as r:
            if r.status != 200:
                return False
        if expected_fingerprint is not None:
            root_url = url.rstrip("/").removesuffix("/v1")
            info = _get_json(f"{root_url}/experiment-info", timeout=timeout)
            return info.get("fingerprint") == expected_fingerprint
        return True
    except Exception:
        return False


def _capture_git_state(repo_root: Path) -> dict:
    out: dict = {}
    try:
        out["commit"] = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        out["branch"] = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        rc = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode
        out["dirty"] = (rc != 0)
    except Exception:
        pass
    return out


def write_manifest(*, spec: dict, spec_path: str, args, urls: list[str],
                   runs: list[dict], root_log: Path,
                   servers: list[dict] | None = None,
                   serving_fingerprint: str | None = None) -> Path:
    import json as _json
    import time as _time
    manifest_path = root_log / "_manifest.json"
    if manifest_path.exists():
        try:
            data = _json.loads(manifest_path.read_text())
        except Exception:
            data = {}
    else:
        data = {}

    previous_fingerprint = data.get("serving_fingerprint")
    if data.get("dispatch_runs") and serving_fingerprint and not previous_fingerprint:
        raise SystemExit(
            "Refusing to resume a matrix whose existing manifest has no "
            "serving fingerprint. Use a new log directory and restart the "
            "entire matrix."
        )
    if (
        previous_fingerprint
        and serving_fingerprint
        and previous_fingerprint != serving_fingerprint
    ):
        raise SystemExit(
            "Refusing to mix serving environments in an existing matrix: "
            f"manifest has {previous_fingerprint}, servers report "
            f"{serving_fingerprint}. Use a new log directory or restart the "
            "entire matrix."
        )

    data.setdefault("title", spec.get("title") or Path(spec_path).stem)
    desc = spec.get("description")
    if desc and "description" not in data:
        data["description"] = desc
    data.setdefault("spec_path", str(spec_path))
    data.setdefault("spec", spec)
    data.setdefault("dispatch_runs", [])
    if serving_fingerprint:
        data.setdefault("serving_fingerprint", serving_fingerprint)
    if servers:
        data.setdefault(
            "reproduction_status",
            servers[0]["experiment_info"].get("reproduction_status"),
        )

    data["dispatch_runs"].append({
        "started_at": _time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_runs": len(runs),
        "n_workers": len(urls),
        "urls": urls,
        "reset_prefix_cache": bool(args.reset_prefix_cache),
        "git": _capture_git_state(REPO_ROOT),
        "serving_fingerprint": serving_fingerprint,
        "servers": servers or [],
    })

    manifest_path.write_text(_json.dumps(data, indent=2, default=str) + "\n")
    return manifest_path


def expand_spec(spec: dict) -> list[dict]:
    base = dict(spec.get("base") or {})
    matrix = spec.get("matrix") or {}
    explicit = spec.get("runs") or []

    for field in list(base) + list(matrix):
        if field not in ALLOWED_FIELDS:
            sys.exit(f"Unknown spec field '{field}'. Allowed: {sorted(ALLOWED_FIELDS)}")

    runs: list[dict] = []
    if matrix:
        keys = list(matrix.keys())
        for combo in itertools.product(*(matrix[k] for k in keys)):
            run = dict(base)
            run.update(dict(zip(keys, combo)))
            runs.append(run)

    for override in explicit:
        for field in override:
            if field not in ALLOWED_FIELDS:
                sys.exit(f"Unknown run field '{field}'.")
        run = dict(base)
        run.update(override)
        runs.append(run)

    for r in runs:
        missing = [f for f in ("agent", "benchmark", "task_id") if f not in r]
        if missing:
            sys.exit(f"Run is missing required fields {missing}: {r}")
    return runs


def run_log_dir(root: Path, run: dict) -> Path:
    seed = run.get("seed", 0)
    agent = run["agent"]
    task = run["task_id"]
    suffix = f"_seed{seed}"
    ms = run.get("memory_strategy")
    if ms and ms != "branching":
        suffix += f"_{ms}"
    rb = run.get("retrieval_backend")
    if rb:
        suffix += f"_{rb}"
    for field, short in _LIBRARIAN_TOGGLE_SHORT.items():
        if run.get(field) is False:
            suffix += f"_no_{short}"
    return root / task / f"{agent}{suffix}"


def already_done(log_dir: Path) -> bool:
    if not log_dir.exists():
        return False
    if any(log_dir.glob("*Evaluator_*.json")):
        return True
    if any(log_dir.glob("*Evaluator_*.partial.json")):
        return True
    return False


def find_partial_in(log_dir: Path) -> Path | None:
    if not log_dir.exists():
        return None
    cands = sorted(
        log_dir.glob("*Evaluator_*.partial.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def find_orphan_partials(plan: list[tuple[int, dict]], root_log: Path) -> list[tuple[dict, Path]]:
    orphans: list[tuple[dict, Path]] = []
    seen: set[Path] = set()
    for _, run in plan:
        ldir = run_log_dir(root_log, run)
        if not ldir.exists():
            continue
        has_final = any(
            not p.name.endswith(".partial.json")
            for p in ldir.glob("*Evaluator_*.json")
        )
        if has_final:
            continue
        partial = find_partial_in(ldir)
        if partial is not None and partial not in seen:
            seen.add(partial)
            orphans.append((run, partial))
    return orphans


def build_cmd(run: dict, url: str | None, log_dir: Path,
              docker_host: str | None = None,
              phase: str = "both",
              partial_path: Path | None = None) -> list[str]:
    base_url = url or "http://invalid/v1"
    cmd = [PYTHON, "-u", str(RUN_PY), "run",
           "--base-url", base_url, "--api-key", "dummy",
           "--log-dir", str(log_dir)]
    if phase != "both":
        cmd += ["--phase", phase]
    if phase == "eval":
        if partial_path is None:
            raise ValueError("phase='eval' requires partial_path")
        cmd += ["--partial-path", str(partial_path)]
    if docker_host:
        cmd += ["--docker-host", docker_host]
    for field, value in run.items():
        if field in _BOOL_FIELDS:
            if value is True:
                cmd += ["--" + field.replace("_", "-")]
            elif value is False:
                cmd += ["--no-" + field.replace("_", "-")]
            continue
        flag = "--" + field.replace("_", "-")
        cmd += [flag, str(value)]
    return cmd


def reset_prefix_cache(url: str, lock: threading.Lock, idx: int) -> bool:
    reset_url = url.rstrip("/").removesuffix("/v1") + "/reset_prefix_cache"
    try:
        req = urllib.request.Request(reset_url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=15.0) as r:
            ok = r.status == 200
    except Exception as e:
        with lock:
            print(f"[RESET-FAIL {idx:>4d}] {reset_url}  err={e}")
        return False
    with lock:
        status = "OK" if ok else f"HTTP-{r.status}"
        print(f"[RESET {idx:>4d}] {reset_url}  {status}")
    return ok


def _run_id(run: dict) -> str:
    rid = f"{run['task_id']}__{run['agent']}__seed{run.get('seed', 0)}"
    ms = run.get("memory_strategy")
    if ms and ms != "branching":
        rid += f"__{ms}"
    rb = run.get("retrieval_backend")
    if rb:
        rid += f"__{rb}"
    for field, short in _LIBRARIAN_TOGGLE_SHORT.items():
        if run.get(field) is False:
            rid += f"__no_{short}"
    return rid


def pick_next_undone(
    plan: list[tuple[int, dict]],
    root_log: Path,
    in_progress: set[str],
    attempts: dict[str, int],
    max_attempts: int,
) -> tuple[int, dict] | None:
    for idx, run in plan:
        rid = _run_id(run)
        if rid in in_progress:
            continue
        if attempts.get(rid, 0) >= max_attempts:
            continue
        ldir = run_log_dir(root_log, run)
        if already_done(ldir):
            continue
        return idx, run
    return None


def _all_idle_and_done(plan, root_log, in_progress, attempts, max_attempts,
                       eval_pending: int = 0) -> bool:
    if in_progress:
        return False
    if eval_pending > 0:
        return False
    return pick_next_undone(
        plan, root_log, in_progress, attempts, max_attempts,
    ) is None


def _run_eval_subprocess(
    run: dict,
    partial_path: Path,
    url: str | None,
    log_dir: Path,
    docker_host: str | None,
    tmp_log_root: Path,
    rid: str,
    state: dict,
    lock: threading.Lock,
    router_release_name: str | None = None,
    router: DispatchRouter | None = None,
) -> int:
    et = run.get("eval_timeout") or 6000
    wall_cap = int(et) + 300

    reserved_server = None
    if (router is not None
            and docker_host is None
            and router_release_name is None):
        reserved_server = router.reserve(task_id=run.get("task_id"))
        docker_host = reserved_server.docker_host

    cmd = build_cmd(
        run, url, log_dir, docker_host=docker_host,
        phase="eval", partial_path=partial_path,
    )
    stdout_log = tmp_log_root / f"{rid}__eval.log"
    with lock:
        host_str = f"  host={reserved_server.name}" if reserved_server else ""
        print(f"[EVAL-START] {rid}{host_str}  partial={partial_path.name}")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, stdout=stdout_log.open("w"),
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    timed_out = False
    try:
        rc = proc.wait(timeout=wall_cap)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            rc = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            rc = proc.wait()
    dt = time.time() - t0

    release_name = router_release_name or (
        reserved_server.name if reserved_server is not None else None
    )
    if router is not None and release_name is not None:
        try:
            router.release(release_name)
        except Exception:
            pass

    counter = state["counter"]
    with lock:
        state["eval_pending"] = max(0, state["eval_pending"] - 1)
        if rc == 0 and not timed_out:
            counter["done"] += 1
            tag = "EVAL-DONE"
        else:
            counter["eval_failed"] = counter.get("eval_failed", 0) + 1
            tag = "EVAL-FAIL" if not timed_out else "EVAL-TOUT"
        print(
            f"[{tag}] {rid}  rc={rc}  {dt/60:.1f}min"
            f"{' (timed out)' if timed_out else ''}  "
            f"eval_pending={state['eval_pending']}"
        )
    return rc


def worker_loop(url: str, plan: list[tuple[int, dict]],
                root_log: Path, tmp_log_root: Path,
                lock: threading.Lock, state: dict,
                reset_before: bool = False,
                router: DispatchRouter | None = None,
                max_attempts: int = 3,
                serving_fingerprint: str | None = None) -> None:
    counter = state["counter"]
    in_progress: set[str] = state["in_progress"]
    attempts: dict[str, int] = state["attempts"]
    unhealthy_urls: set[str] = state["unhealthy_urls"]
    unhealthy_since: float | None = None

    while True:
        if state.get("fatal_error"):
            return
        if not is_url_healthy(url, expected_fingerprint=serving_fingerprint):
            if serving_fingerprint is not None:
                try:
                    current = probe_server(url)["experiment_info"].get("fingerprint")
                except Exception:
                    current = None
                if current is not None and current != serving_fingerprint:
                    message = (
                        f"Serving fingerprint changed during the matrix at {url}: "
                        f"expected {serving_fingerprint}, found {current}. Stop and "
                        "restart the entire matrix in a new log directory."
                    )
                    with lock:
                        state["fatal_error"] = message
                        unhealthy_urls.add(url)
                        print(f"[FATAL] {message}")
                    return
            now = time.time()
            if unhealthy_since is None:
                unhealthy_since = now
                with lock:
                    unhealthy_urls.add(url)
                    print(f"[UNHEALTHY] {url}  — worker idle until URL returns")
            with lock:
                if _all_idle_and_done(
                    plan, root_log, in_progress, attempts, max_attempts,
                    eval_pending=state.get("eval_pending", 0),
                ):
                    print(f"[BAILOUT ] {url}  — URL still unhealthy but "
                          f"matrix complete; exiting worker")
                    return
            time.sleep(30.0)
            continue
        elif unhealthy_since is not None:
            with lock:
                unhealthy_urls.discard(url)
                print(f"[RECOVER ] {url}  — back online after "
                      f"{time.time()-unhealthy_since:.0f}s")
            unhealthy_since = None

        with lock:
            picked = pick_next_undone(
                plan, root_log, in_progress, attempts, max_attempts,
            )
            if picked is None:
                if not in_progress and state.get("eval_pending", 0) == 0:
                    return
                pass
            else:
                idx, run = picked
                rid = _run_id(run)
                in_progress.add(rid)
                attempts[rid] = attempts.get(rid, 0) + 1
                attempt_n = attempts[rid]
        if picked is None:
            time.sleep(2.0)
            continue

        ldir = run_log_dir(root_log, run)
        ldir.mkdir(parents=True, exist_ok=True)
        stdout_log = tmp_log_root / f"{rid}__{url_slug(url)}__a{attempt_n}.log"

        if reset_before:
            if not reset_prefix_cache(url, lock, idx):
                with lock:
                    in_progress.discard(rid)
                    counter["failed"] += 1
                    print(f"[ABORT {idx:>4d}] {rid}  reset failed")
                continue

        server = None
        docker_host = None
        if router is not None:
            server = router.reserve(task_id=run["task_id"])
            docker_host = server.docker_host
        eval_pool: ThreadPoolExecutor | None = state.get("eval_pool")
        run_phase = "agent"  # two-phase is mandatory; eval handed to eval_pool

        with lock:
            host_str = f"  host={server.name}" if server else ""
            print(f"[START {idx:>4d}] {rid}  url={url}{host_str}  "
                  f"attempt={attempt_n}/{max_attempts}  "
                  f"phase={run_phase}  log={stdout_log}")
        cmd = build_cmd(run, url, ldir, docker_host=docker_host, phase=run_phase)
        rt = run.get("run_timeout")
        et = run.get("eval_timeout") or 6000
        if run_phase == "agent":
            wall_cap = (int(rt) + 300) if rt else None
        else:
            wall_cap = (int(rt) + int(et) + 300) if rt else None
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=stdout_log.open("w"),
                                stderr=subprocess.STDOUT,
                                start_new_session=True)
        timed_out = False
        try:
            rc = proc.wait(timeout=wall_cap)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
                rc = proc.wait()
        dt = time.time() - t0

        if router is not None and server is not None:
            router.release(server.name)

        url_dead_after = (
            rc != 0
            and rc != 3
            and not is_url_healthy(
                url, expected_fingerprint=serving_fingerprint,
            )
        )

        queued_eval = False
        if (run_phase == "agent" and rc == 0 and not timed_out):
            partial = find_partial_in(ldir)
            if partial is None:
                rc = 1
            else:
                with lock:
                    state["eval_pending"] = state.get("eval_pending", 0) + 1
                eval_pool.submit(
                    _run_eval_subprocess,
                    run=run,
                    partial_path=partial,
                    url=url,
                    log_dir=ldir,
                    docker_host=None,
                    tmp_log_root=tmp_log_root,
                    rid=rid,
                    state=state,
                    lock=lock,
                    router_release_name=None,
                    router=router,
                )
                queued_eval = True

        with lock:
            in_progress.discard(rid)
            if queued_eval:
                host_str = f"  host={server.name}" if server else ""
                print(
                    f"[AGENT-DONE {idx:>4d}] {rid}  url={url}{host_str}  "
                    f"rc=0  attempt={attempt_n}/{max_attempts}  "
                    f"{dt/60:.1f}min  → eval queued "
                    f"(pending={state['eval_pending']})"
                )
                continue
            if timed_out:
                tag = "TOUT "
                counter["failed"] += 1
            elif rc == 0:
                tag = "DONE "
                counter["done"] += 1
            elif rc == 3:
                tag = "INFRA"
                counter["infra_skip"] += 1
            elif url_dead_after:
                tag = "URLFAIL"
                counter["url_failed"] = counter.get("url_failed", 0) + 1
                attempts[rid] = max(0, attempts[rid] - 1)
            else:
                tag = "FAIL "
                counter["failed"] += 1
            host_str = f"  host={server.name}" if server else ""
            print(f"[{tag} {idx:>4d}] {rid}  url={url}{host_str}  "
                  f"rc={rc}  attempt={attempt_n}/{max_attempts}  "
                  f"{dt/60:.1f}min{' (timed out)' if timed_out else ''}")


def url_slug(url: str) -> str:
    return (url.replace("http://", "").replace("https://", "")
               .replace("/", "_").replace(":", "-"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, help="Matrix YAML path")
    ap.add_argument("--urls", help="Comma-separated vLLM URLs")
    ap.add_argument("--urls-file", help="File with one vLLM URL per line")
    ap.add_argument("--log-dir", help="Override spec.log_dir")
    ap.add_argument("--tmp-log-dir", default="/tmp/run_matrix_runs")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print expansion and partitioning, then exit")
    ap.add_argument("--no-probe", action="store_true",
                    help="Skip server probes for --dry-run only. Formal runs "
                         "always enforce serving fingerprints.")
    ap.add_argument("--reset-prefix-cache", action="store_true",
                    help="POST /reset_prefix_cache before each run (requires "
                         "vLLM launched with VLLM_SERVER_DEV_MODE=1). Prevents "
                         "prior-episode prefix-cache hits from polluting "
                         "per-turn joules measurements.")
    ap.add_argument("--server-pool", default=None,
                    help="Path to a server-pool YAML for distributed docker "
                         "eval across remote daemons. Default: unset = local "
                         "docker only (no pool needed for a single machine).")
    ap.add_argument("--no-distributed", action="store_true",
                    help="Disable docker server pool routing — every run "
                         "uses the local docker daemon. Useful when "
                         "running on a single machine without ssh-pool.")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="Per-run cap on dispatcher-side retries within "
                         "one session. Bounds infra-failure loops (e.g. "
                         "docker image truly missing from registry). "
                         "Default 3.")
    ap.add_argument("--eval-pool-size", type=int, default=4,
                    help="Phase-B (eval) thread count (must be >=1). The "
                         "matrix ALWAYS runs two-phase: workers run --phase "
                         "agent and hand --phase eval to a ThreadPoolExecutor "
                         "of this size, which overlaps eval with the next "
                         "agent's vLLM-bound work and throttles concurrent "
                         "docker churn. Single-shot inline eval has been "
                         "removed. Recommend 4–8 on a 24-core box; 1–2 to "
                         "reduce docker host load.")
    args = ap.parse_args()

    if args.eval_pool_size < 1:
        sys.exit(
            "--eval-pool-size must be >= 1: the agent/eval phase split is "
            "mandatory (single-shot inline eval has been removed)."
        )

    spec = yaml.safe_load(Path(args.spec).read_text())
    runs = expand_spec(spec)
    urls = load_urls(args)
    if args.no_probe and not args.dry_run:
        sys.exit("--no-probe is allowed only with --dry-run")

    matrix_models = {r.get("model") for r in runs if r.get("model")}
    if len(matrix_models) > 1:
        sys.exit(
            "A matrix dispatched to one server pool must use one model; found: "
            + ", ".join(sorted(matrix_models))
        )
    expected_model = next(iter(matrix_models), None)
    servers: list[dict] = []
    serving_fingerprint: str | None = None
    if not args.no_probe:
        for url in urls:
            try:
                server = probe_server(url)
            except Exception as exc:
                sys.exit(
                    f"Serving preflight failed for {url}: {type(exc).__name__}: "
                    f"{exc}. Formal dispatch requires /v1/models and "
                    "/experiment-info."
                )
            servers.append(server)
            info = server["experiment_info"]
            print(
                f"[OK] {url} -> model={info.get('model')} "
                f"fingerprint={info.get('fingerprint')}"
            )
        try:
            serving_fingerprint = validate_server_pool(servers, expected_model)
        except ValueError as exc:
            sys.exit(str(exc))

    title = spec.get("title") or Path(args.spec).stem
    root_log = Path(args.log_dir or spec.get("log_dir") or f"logs/{title}")
    root_log.mkdir(parents=True, exist_ok=True)
    tmp_log = Path(args.tmp_log_dir) / title
    tmp_log.mkdir(parents=True, exist_ok=True)

    plan: list[tuple[int, dict]] = [(i, r) for i, r in enumerate(runs)]

    router: DispatchRouter | None = None
    if not args.no_distributed:
        try:
            router = DispatchRouter(args.server_pool) if args.server_pool else DispatchRouter()
            if not router.enabled:
                router = None
        except FileNotFoundError as e:
            print(f"[WARN] server pool config missing ({e}); running local-only")
            router = None

    print(f"=== {title} ===")
    print(f"  runs:      {len(runs)}")
    print(f"  workers:   {len(urls)} ({', '.join(urls)})")
    print(f"  log_dir:   {root_log}")
    print(f"  tmp_log:   {tmp_log}")
    print(f"  picker:    disk-driven re-scan (auto-retries missing logs)")
    print(f"  max attempts/run: {args.max_attempts}")
    if router is not None:
        print(f"  servers:   {len(router.servers)} pool, {router.total_slots} total slots")
        for name, srv in sorted(router.servers.items()):
            print(f"    - {name}: max_concurrent={srv.max_concurrent} "
                  f"disk_budget={srv.disk_budget_gb}G")
    else:
        print(f"  servers:   (local docker only)")

    if args.dry_run:
        for i, r in enumerate(runs):
            print(f"  run[{i}] (any-url) {r}")
        return 0

    manifest_path = write_manifest(
        spec=spec, spec_path=args.spec, args=args,
        urls=urls, runs=runs, root_log=root_log,
        servers=servers, serving_fingerprint=serving_fingerprint,
    )
    print(f"  manifest:  {manifest_path}")

    lock = threading.Lock()
    state = {
        "counter": {"done": 0, "failed": 0, "infra_skip": 0, "url_failed": 0,
                    "eval_failed": 0},
        "in_progress": set(),
        "attempts": {},
        "unhealthy_urls": set(),
        "eval_pending": 0,
        "eval_pool": None,
        "fatal_error": None,
    }
    # Two-phase (agent → eval pool) is mandatory; --eval-pool-size is
    # validated >= 1 above, so the eval pool is always built.
    eval_pool: ThreadPoolExecutor = ThreadPoolExecutor(
        max_workers=args.eval_pool_size, thread_name_prefix="eval",
    )
    state["eval_pool"] = eval_pool
    orphans = find_orphan_partials(plan, root_log)
    if orphans:
        print(f"  resuming {len(orphans)} orphan partial(s) "
              f"into the eval pool")
        for run, partial in orphans:
            rid = _run_id(run)
            ldir = run_log_dir(root_log, run)
            with lock:
                state["eval_pending"] += 1
            eval_pool.submit(
                _run_eval_subprocess,
                run=run,
                partial_path=partial,
                url=urls[0] if urls else None,
                log_dir=ldir,
                docker_host=None,
                tmp_log_root=tmp_log,
                rid=rid,
                state=state,
                lock=lock,
                router_release_name=None,
                router=None,
            )
    print(f"  eval pool: {args.eval_pool_size} thread(s)")

    threads = []
    for u in urls:
        t = threading.Thread(
            target=worker_loop,
            args=(u, plan, root_log, tmp_log, lock, state,
                  args.reset_prefix_cache, router, args.max_attempts,
                  serving_fingerprint),
            daemon=False,
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    if eval_pool is not None:
        eval_pool.shutdown(wait=True)

    if state.get("fatal_error"):
        sys.exit(state["fatal_error"])

    c = state["counter"]
    print(f"\n=== DONE ===  done={c['done']}  "
          f"failed={c['failed']}  infra_skip={c['infra_skip']}  "
          f"url_failed={c['url_failed']}  "
          f"eval_failed={c.get('eval_failed', 0)}")
    still_missing = sum(
        1 for _, r in plan if not already_done(run_log_dir(root_log, r))
    )
    if still_missing:
        print(f"  still missing logs: {still_missing} runs "
              f"(max_attempts={args.max_attempts} exhausted)")
    return 0 if c["failed"] == 0 and still_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
