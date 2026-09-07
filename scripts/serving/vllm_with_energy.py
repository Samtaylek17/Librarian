from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time

import pynvml


def _init_nvml():
    pynvml.nvmlInit()
    count = pynvml.nvmlDeviceGetCount()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd.strip():
        try:
            indices = [int(x) for x in cvd.split(",") if x.strip().isdigit()]
        except Exception:
            indices = list(range(count))
        indices = [i for i in indices if 0 <= i < count]
        if not indices:
            indices = list(range(count))
    else:
        indices = list(range(count))
    return [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices]


_HANDLES: list = []
_EXPERIMENT_INFO: dict = {}


def _decode_nvml(value):
    return value.decode() if isinstance(value, bytes) else value


def _option_value(argv: list[str], name: str) -> str | None:
    """Return argparse's effective (last) value for an option."""
    value = None
    for idx, arg in enumerate(argv):
        if arg == name and idx + 1 < len(argv):
            value = argv[idx + 1]
        elif arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
    return value


def _serve_model(argv: list[str]) -> str | None:
    try:
        idx = argv.index("serve")
    except ValueError:
        return None
    if idx + 1 >= len(argv) or argv[idx + 1].startswith("-"):
        return None
    return argv[idx + 1]


def _version_tuple(version: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", version)
    return tuple(int(x) for x in match.group(1).split(".")) if match else ()


def _registered_tool_parsers() -> set[str]:
    candidates = (
        "vllm.tool_parsers",
        "vllm.tool_parsers.abstract_tool_parser",
        "vllm.entrypoints.openai.tool_parsers",
        "vllm.entrypoints.openai.tool_parsers.abstract_tool_parser",
    )
    for module_name in candidates:
        try:
            module = __import__(module_name, fromlist=["ToolParserManager"])
            manager = getattr(module, "ToolParserManager")
            registry = getattr(manager, "tool_parsers", None)
            if registry is None:
                registry = getattr(manager, "_tool_parsers", None)
            if isinstance(registry, dict):
                lazy_registry = getattr(manager, "lazy_parsers", {})
                return set(registry) | (
                    set(lazy_registry) if isinstance(lazy_registry, dict) else set()
                )
        except (ImportError, AttributeError):
            continue
    raise RuntimeError("could not inspect vLLM's registered tool parsers")


def _validate_preflight(
    *,
    argv: list[str],
    environ: dict[str, str],
    vllm_version: str,
    tool_parsers: set[str],
    visible_gpu_count: int,
) -> dict:
    """Validate the immutable serving contract before vLLM allocates VRAM."""
    minimum = environ.get("VLLM_MIN_VERSION", "0.19.0")
    expected_version = environ.get("VLLM_EXPECTED_VERSION", "0.19.1")
    expected_parser = environ.get("VLLM_EXPECTED_TOOL_PARSER", "qwen3_xml")
    expected_model = environ.get("VLLM_EXPECTED_MODEL")
    expected_revision = environ.get("VLLM_MODEL_REVISION")
    expected_gpus = int(environ.get("VLLM_EXPECTED_GPU_COUNT", "1"))

    model = _serve_model(argv)
    parser = _option_value(argv, "--tool-call-parser")
    revision = _option_value(argv, "--revision")
    tokenizer_revision = _option_value(argv, "--tokenizer-revision")

    errors: list[str] = []
    if _version_tuple(vllm_version) < _version_tuple(minimum):
        errors.append(f"vLLM {vllm_version} is older than required {minimum}")
    if expected_version and vllm_version != expected_version:
        errors.append(
            f"vLLM version {vllm_version!r} does not match the locked "
            f"version {expected_version!r}"
        )
    if parser != expected_parser:
        errors.append(
            f"tool parser {parser!r} does not match locked parser "
            f"{expected_parser!r}"
        )
    if expected_parser not in tool_parsers:
        errors.append(
            f"locked parser {expected_parser!r} is unavailable; registered: "
            + ", ".join(sorted(tool_parsers))
        )
    if expected_model and model != expected_model:
        errors.append(
            f"served model {model!r} does not match locked model "
            f"{expected_model!r}"
        )
    if not expected_revision:
        errors.append("VLLM_MODEL_REVISION must contain a full checkpoint SHA")
    else:
        if not re.fullmatch(r"[0-9a-f]{40}", expected_revision):
            errors.append("VLLM_MODEL_REVISION must be a full 40-character SHA")
        if revision != expected_revision:
            errors.append(
                f"model revision {revision!r} does not match locked revision "
                f"{expected_revision!r}"
            )
        if tokenizer_revision != expected_revision:
            errors.append(
                "tokenizer revision must equal the locked model revision "
                f"{expected_revision!r}"
            )
    if visible_gpu_count != expected_gpus:
        errors.append(
            f"expected {expected_gpus} visible GPU, found {visible_gpu_count}"
        )
    if errors:
        raise RuntimeError("serving preflight failed:\n  - " + "\n  - ".join(errors))
    return {
        "model": model,
        "model_revision": expected_revision,
        "tokenizer_revision": tokenizer_revision,
        "tool_call_parser": parser,
        "vllm_version": vllm_version,
    }


def _safe_package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _redacted_argv(argv: list[str]) -> list[str]:
    sensitive_options = {
        "--access-token",
        "--api-key",
        "--auth-token",
        "--hf-token",
        "--password",
        "--secret",
        "--token",
    }
    redacted: list[str] = []
    redact_next = False
    for arg in argv:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        option = arg.split("=", 1)[0].lower()
        if option in sensitive_options:
            if "=" in arg:
                redacted.append(arg.split("=", 1)[0] + "=<redacted>")
            else:
                redacted.append(arg)
                redact_next = True
        else:
            redacted.append(arg)
    return redacted


def _fingerprint_argv(argv: list[str]) -> list[str]:
    """Drop endpoint-local values while retaining inference behavior."""
    result: list[str] = []
    skip_next = False
    for arg in _redacted_argv(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--port":
            skip_next = True
            continue
        if arg.startswith("--port="):
            continue
        result.append(arg)
    return result


def _nvml_call(name: str, handle=None, *args):
    try:
        func = getattr(pynvml, name)
        value = func(handle, *args) if handle is not None else func(*args)
        return _decode_nvml(value)
    except (AttributeError, pynvml.NVMLError):
        return None


def _gpu_info(handle) -> dict:
    memory = _nvml_call("nvmlDeviceGetMemoryInfo", handle)
    clocks = {}
    applications_clocks = {}
    for label, constant in (
        ("graphics", getattr(pynvml, "NVML_CLOCK_GRAPHICS", None)),
        ("sm", getattr(pynvml, "NVML_CLOCK_SM", None)),
        ("memory", getattr(pynvml, "NVML_CLOCK_MEM", None)),
    ):
        if constant is not None:
            clocks[label] = _nvml_call("nvmlDeviceGetClockInfo", handle, constant)
            applications_clocks[label] = _nvml_call(
                "nvmlDeviceGetApplicationsClock", handle, constant,
            )
    power_limit = _nvml_call("nvmlDeviceGetPowerManagementLimit", handle)
    return {
        "uuid": _nvml_call("nvmlDeviceGetUUID", handle),
        "name": _nvml_call("nvmlDeviceGetName", handle),
        "memory_total_bytes": getattr(memory, "total", None),
        "power_limit_w": power_limit / 1000.0 if power_limit is not None else None,
        "current_clocks_mhz": clocks,
        "applications_clocks_mhz": applications_clocks,
    }


def _build_experiment_info(contract: dict, argv: list[str]) -> dict:
    try:
        import torch
        torch_cuda = torch.version.cuda
    except Exception:
        torch_cuda = None

    gpus = [_gpu_info(handle) for handle in _HANDLES]
    comparable = {
        "reproduction_status": os.environ.get(
            "LIBRARIAN_REPRODUCTION_STATUS",
            "reconstructed reproduction using vLLM 0.19.1",
        ),
        "image_digest": os.environ.get("VLLM_IMAGE_DIGEST"),
        "repository_commit": os.environ.get("LIBRARIAN_GIT_COMMIT"),
        "repository_dirty": os.environ.get("LIBRARIAN_GIT_DIRTY") == "1",
        "cuda_compatibility_enabled": os.environ.get(
            "VLLM_ENABLE_CUDA_COMPATIBILITY", "0",
        ).lower() in {"1", "true"},
        "model": contract["model"],
        "model_revision": contract["model_revision"],
        "tokenizer_revision": contract["tokenizer_revision"],
        "tool_call_parser": contract["tool_call_parser"],
        "software": {
            "vllm": contract["vllm_version"],
            "torch": _safe_package_version("torch"),
            "transformers": _safe_package_version("transformers"),
            "cuda_runtime": torch_cuda,
        },
        "driver_version": _nvml_call("nvmlSystemGetDriverVersion"),
        "gpu_contract": [
            {
                "name": gpu["name"],
                "memory_total_bytes": gpu["memory_total_bytes"],
                "power_limit_w": gpu["power_limit_w"],
                "applications_clocks_mhz": gpu["applications_clocks_mhz"],
            }
            for gpu in gpus
        ],
        "launch_argv": _fingerprint_argv(argv),
    }
    encoded = json.dumps(comparable, sort_keys=True, separators=(",", ":"))
    return {
        **comparable,
        "fingerprint": hashlib.sha256(encoded.encode()).hexdigest(),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "launch_argv_full": _redacted_argv(argv),
    }


def _energy_snapshot() -> dict:
    gpus_mJ = []
    gpus_power_W = []
    for h in _HANDLES:
        try:
            gpus_mJ.append(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
        except pynvml.NVMLError:
            gpus_mJ.append(None)
        try:
            gpus_power_W.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
        except pynvml.NVMLError:
            gpus_power_W.append(None)
    return {
        "t": time.time(),
        "gpus_mJ": gpus_mJ,
        "gpus_power_W": gpus_power_W,
        "n_gpus": len(_HANDLES),
    }


def _format_mJ(snap: dict) -> str:
    return ",".join(
        str(x) if x is not None else "" for x in snap["gpus_mJ"]
    )


_ENERGY_PATHS = (
    "/v1/chat/completions",
    "/v1/completions",
)


async def _energy_dispatch(request, call_next):
    if request.url.path not in _ENERGY_PATHS:
        return await call_next(request)
    e_start = _energy_snapshot()
    response = await call_next(request)
    e_end = _energy_snapshot()
    response.headers["X-Energy-T-Start"] = f"{e_start['t']:.6f}"
    response.headers["X-Energy-T-End"] = f"{e_end['t']:.6f}"
    response.headers["X-Energy-MJ-Start"] = _format_mJ(e_start)
    response.headers["X-Energy-MJ-End"] = _format_mJ(e_end)
    response.headers["X-Energy-N-GPUs"] = str(e_end["n_gpus"])
    return response


def _install_energy_route():
    from vllm.entrypoints.openai import api_server as vllm_api

    hook_target = None
    for name in ("build_app", "create_app", "init_app"):
        if hasattr(vllm_api, name):
            hook_target = name
            break

    if hook_target is None:
        available = [x for x in dir(vllm_api) if "app" in x.lower()]
        raise RuntimeError(
            "Could not find a vLLM app-builder symbol to patch. "
            f"Tried build_app/create_app/init_app. Available app-ish symbols: {available}. "
            "Please update scripts/serving/vllm_with_energy.py with the new hook."
        )

    original = getattr(vllm_api, hook_target)

    def patched(*args, **kwargs):
        app = original(*args, **kwargs)
        installed = []
        if hasattr(app, "add_api_route"):
            app.add_api_route("/energy", _energy_snapshot, methods=["GET"])
            app.add_api_route(
                "/experiment-info", lambda: _EXPERIMENT_INFO, methods=["GET"],
            )
            installed.append("GET /energy + GET /experiment-info")
        if hasattr(app, "add_middleware"):
            try:
                from starlette.middleware.base import BaseHTTPMiddleware
                if hasattr(app, "middleware_stack"):
                    app.middleware_stack = None
                app.add_middleware(
                    BaseHTTPMiddleware, dispatch=_energy_dispatch,
                )
                installed.append("X-Energy-* response headers")
            except Exception as e:
                print(
                    f"[energy-probe] WARNING: middleware install failed: {e!r}",
                    flush=True,
                )
        if installed:
            print(
                f"[energy-probe] installed via {hook_target}: "
                + ", ".join(installed),
                flush=True,
            )
        else:
            print(
                f"[energy-probe] WARNING: {hook_target} returned {type(app)!r} "
                "with no FastAPI hooks; energy capture DISABLED",
                flush=True,
            )
        return app

    setattr(vllm_api, hook_target, patched)


def main() -> None:
    global _HANDLES, _EXPERIMENT_INFO
    _HANDLES = _init_nvml()
    print(f"[energy-probe] NVML initialized, {len(_HANDLES)} GPU(s) visible", flush=True)

    vllm_version = importlib.metadata.version("vllm")
    contract = _validate_preflight(
        argv=sys.argv[1:],
        environ=dict(os.environ),
        vllm_version=vllm_version,
        tool_parsers=_registered_tool_parsers(),
        visible_gpu_count=len(_HANDLES),
    )
    _EXPERIMENT_INFO = _build_experiment_info(contract, sys.argv[1:])
    print(
        "[energy-probe] serving contract verified: "
        f"fingerprint={_EXPERIMENT_INFO['fingerprint']}",
        flush=True,
    )

    _install_energy_route()

    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()


if __name__ == "__main__":
    main()
