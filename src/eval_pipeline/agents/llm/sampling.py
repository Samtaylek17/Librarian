from __future__ import annotations

from typing import Any

from eval_pipeline.agents.llm.profiles import get_profile


SAMPLING_DEFAULTS: dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}


def pop_sampling_kwargs(
    kwargs: dict[str, Any],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    preset = get_profile(model).sampling
    return {
        k: kwargs.pop(k, preset.get(k, default))
        for k, default in SAMPLING_DEFAULTS.items()
    }
