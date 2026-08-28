from __future__ import annotations

import sys
import os
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
            installed.append("GET /energy")
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
    global _HANDLES
    _HANDLES = _init_nvml()
    print(f"[energy-probe] NVML initialized, {len(_HANDLES)} GPU(s) visible", flush=True)

    _install_energy_route()

    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()


if __name__ == "__main__":
    main()
