#!/usr/bin/env python3
"""Non-reportable preflight for a pinned Librarian vLLM worker."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any


def _request(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> tuple[dict, Any]:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            parsed = json.loads(raw) if raw else {}
            return parsed, response.headers
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url} returned {exc.code}: {detail}") from exc


def _energy_values(value: str | None) -> list[float]:
    if value is None:
        raise AssertionError("missing energy header")
    return [float(item) for item in value.split(",")]


def _check_energy_headers(headers: Any) -> None:
    if headers.get("X-Energy-N-GPUs") != "1":
        raise AssertionError("completion did not report exactly one energy GPU")
    starts = _energy_values(headers.get("X-Energy-MJ-Start"))
    ends = _energy_values(headers.get("X-Energy-MJ-End"))
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise AssertionError(f"non-monotonic completion energy: {starts} -> {ends}")


def _chat(base_url: str, model: str, payload: dict) -> dict:
    request = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 128,
        **payload,
    }
    response, headers = _request(
        f"{base_url}/chat/completions", method="POST", body=request,
    )
    _check_energy_headers(headers)
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0].get("message"), dict):
        raise AssertionError(f"invalid chat-completion response: {response}")
    return response


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Return a test value for {name}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                },
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    }


def _parsed_tool_calls(message: dict) -> list[dict]:
    calls = message.get("tool_calls") or []
    for call in calls:
        function = call.get("function") or {}
        if not function.get("name"):
            raise AssertionError(f"tool call has no function name: {call}")
        args = function.get("arguments")
        if isinstance(args, str):
            json.loads(args)
        elif not isinstance(args, dict):
            raise AssertionError(f"tool arguments are not JSON-compatible: {args!r}")
    return calls


def validate(base_url: str, expected_model: str | None) -> dict:
    base_url = base_url.rstrip("/")
    root_url = base_url.removesuffix("/v1")
    info, _ = _request(f"{root_url}/experiment-info")
    models, _ = _request(f"{base_url}/models")
    model_ids = [item.get("id") for item in models.get("data") or []]
    model = expected_model or info.get("model")

    if not info.get("fingerprint"):
        raise AssertionError("/experiment-info has no fingerprint")
    if info.get("gpu_count") != 1:
        raise AssertionError(f"server exposes {info.get('gpu_count')} GPUs, expected 1")
    if info.get("model") != model or model not in model_ids:
        raise AssertionError(
            f"model mismatch: expected={model!r}, info={info.get('model')!r}, "
            f"advertised={model_ids!r}"
        )

    first_energy, _ = _request(f"{root_url}/energy")
    ordinary = _chat(
        base_url,
        model,
        {"messages": [{"role": "user", "content": "Reply with only: ready"}]},
    )
    if not ordinary["choices"][0]["message"].get("content"):
        raise AssertionError("ordinary completion returned no content")

    single = _chat(
        base_url,
        model,
        {
            "messages": [{
                "role": "user",
                "content": "Call probe_alpha exactly once with key alpha. Do not answer in text.",
            }],
            "tools": [_tool("probe_alpha")],
            "tool_choice": "required",
        },
    )
    single_message = single["choices"][0]["message"]
    single_calls = _parsed_tool_calls(single_message)
    if len(single_calls) != 1:
        raise AssertionError(f"expected one parsed tool call, received {single_calls!r}")
    if not single_calls[0].get("id"):
        raise AssertionError("parsed tool call has no call ID")

    roundtrip_messages = [
        {
            "role": "user",
            "content": "Call probe_alpha exactly once with key alpha. Do not answer in text.",
        },
        {
            "role": "assistant",
            "content": single_message.get("content"),
            "tool_calls": single_calls,
        },
        {
            "role": "tool",
            "tool_call_id": single_calls[0]["id"],
            "content": json.dumps({"value": "alpha-ok"}),
        },
    ]
    roundtrip = _chat(base_url, model, {"messages": roundtrip_messages})
    if not roundtrip["choices"][0]["message"].get("content"):
        raise AssertionError("tool-result round trip returned no final content")

    multiple = _chat(
        base_url,
        model,
        {
            "messages": [{
                "role": "user",
                "content": (
                    "Call both probe_alpha with key alpha and probe_beta with key beta "
                    "before answering. Emit exactly two tool calls and no text."
                ),
            }],
            "tools": [_tool("probe_alpha"), _tool("probe_beta")],
            "tool_choice": "required",
        },
    )
    multiple_calls = _parsed_tool_calls(multiple["choices"][0]["message"])
    if len(multiple_calls) != 2:
        raise AssertionError(f"expected two parsed tool calls, received {multiple_calls!r}")

    last_energy, _ = _request(f"{root_url}/energy")
    starts = first_energy.get("gpus_mJ") or []
    ends = last_energy.get("gpus_mJ") or []
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        raise AssertionError(f"energy counter did not increase: {starts} -> {ends}")

    _request(f"{root_url}/reset_prefix_cache", method="POST")
    return {
        "ok": True,
        "model": model,
        "fingerprint": info["fingerprint"],
        "reproduction_status": info.get("reproduction_status"),
        "checks": [
            "ordinary_completion",
            "single_tool_call",
            "tool_result_round_trip",
            "multiple_tool_calls",
            "energy_headers",
            "energy_counter",
            "prefix_cache_reset",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--expected-model")
    args = parser.parse_args()
    try:
        result = validate(args.base_url, args.expected_model)
    except Exception as exc:
        root_url = args.base_url.rstrip("/").removesuffix("/v1")
        try:
            _request(f"{root_url}/reset_prefix_cache", method="POST")
        except Exception:
            pass
        print(f"VALIDATION FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    print("Pilot validation passed. Keep this output separate from formal results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
