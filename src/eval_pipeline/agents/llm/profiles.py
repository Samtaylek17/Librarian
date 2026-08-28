from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from eval_pipeline.agents.llm.tool_format import (
    ToolCallRenderer,
    identity_renderer,
    seed_oss_renderer,
)


@dataclass(frozen=True)
class ModelProfile:

    sampling: dict[str, Any] = field(default_factory=dict)

    reasoning_block_pattern: re.Pattern | None = None

    thinking_extra_body: dict = field(default_factory=dict)

    no_thinking_extra_body: dict = field(default_factory=dict)

    tool_call_renderer: ToolCallRenderer = identity_renderer


_QWEN3 = ModelProfile(
    sampling={
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
    reasoning_block_pattern=re.compile(r"<think>.*?</think>", re.DOTALL),
    thinking_extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    no_thinking_extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)


_SEED_OSS = ModelProfile(
    sampling={
        "temperature": 1.1,
        "top_p": 0.95,
        "top_k": -1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    },
    reasoning_block_pattern=re.compile(
        r"<seed:think>.*?</seed:think>", re.DOTALL,
    ),
    thinking_extra_body={"chat_template_kwargs": {"thinking_budget": -1}},
    no_thinking_extra_body={"chat_template_kwargs": {"thinking_budget": 0}},
    tool_call_renderer=seed_oss_renderer,
)


_DEFAULT = ModelProfile()


_REGISTRY: list[tuple[re.Pattern, ModelProfile]] = [
    (re.compile(r"qwen3", re.I), _QWEN3),
    (re.compile(r"seed-?oss", re.I), _SEED_OSS),
]


def get_profile(model: str | None) -> ModelProfile:
    if not model:
        return _DEFAULT
    for name_re, profile in _REGISTRY:
        if name_re.search(model):
            return profile
    return _DEFAULT


def strip_reasoning(text: str, model: str | None) -> str:
    if not text:
        return text
    profile = get_profile(model)
    if profile.reasoning_block_pattern is None:
        return text.strip()
    return profile.reasoning_block_pattern.sub("", text).strip()


def thinking_extra_body(model: str | None, *, on: bool = True) -> dict:
    profile = get_profile(model)
    src = profile.thinking_extra_body if on else profile.no_thinking_extra_body
    return copy.deepcopy(src)


def strip_subagent_output(result, model: str | None) -> str:
    return strip_reasoning(getattr(result, "final_output", None) or "", model)


def render_tool_examples(text: str, model: str | None) -> str:
    return get_profile(model).tool_call_renderer(text)
