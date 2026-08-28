"""Bare OpenAI-API-compatible backend (Azure OpenAI, OpenRouter, …).

Same wire-protocol shape as :class:`VllmLLM` but without the
vLLM-specific knobs:

  - No ``extra_body`` defaults (``top_k``, ``min_p``, ``repetition_penalty``,
    ``chat_template_kwargs``) — most non-vLLM hosts reject unknown
    fields there.
  - No ``/energy`` endpoint snapshot.
  - Schema quirks are opt-in (default ``None``); the Qwen anyOf+null
    bug is vLLM-specific and doesn't apply.

Use this for:
  - Azure OpenAI (``base_url='https://...openai.azure.com/...'``)
  - OpenRouter (``base_url='https://openrouter.ai/api/v1'``)
  - bare openai.com (``base_url=None``)
  - any chat-completions-compatible host that doesn't accept vLLM extras

For vLLM: keep using :class:`VllmLLM` — its extra_body knobs are
load-bearing (top_k, min_p, repetition_penalty are the Qwen
precise-coding preset).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from eval_pipeline.agents.llm.quirks import SchemaQuirk


class OpenAICompatLLM:

    def __init__(
        self,
        model: str,
        openai_client,
        *,
        seed: int | None = None,
        tokenizer: Any = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        tool_choice: str | None = None,
        live_http_log_path: str | None = None,
        quirk: SchemaQuirk | None = None,
    ) -> None:
        from agents.models.interface import Model
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

        self._inner = OpenAIChatCompletionsModel(
            model=model, openai_client=openai_client,
        )
        self._seed = seed
        self._openai_client = openai_client
        self._model = model
        self._tokenizer = tokenizer
        self._temperature = temperature
        self._top_p = top_p
        self._presence_penalty = presence_penalty
        self._frequency_penalty = frequency_penalty
        self._tool_choice = tool_choice
        self._quirk = quirk
        Model.register(OpenAICompatLLM)
        self._live_http_log_path = live_http_log_path
        if live_http_log_path:
            try:
                os.makedirs(os.path.dirname(live_http_log_path), exist_ok=True)
                open(live_http_log_path, "w").close()
            except Exception:
                pass
        self._wrap_chat_completions_for_logging()


    def _wrap_chat_completions_for_logging(self) -> None:
        if not self._live_http_log_path:
            return
        client = self._openai_client
        if getattr(client, "_swe_logging_wrapped", False):
            return
        orig = client.chat.completions.create

        async def _logged_create(*args, **kwargs):
            from eval_pipeline.agents.runtime.agent_registry import (
                enforce_llm_agent_gate,
            )

            # Requirement 1: only agents may call the LLM.
            enforce_llm_agent_gate()

            t0 = time.time()
            captured_exc: Exception | None = None
            resp = None
            try:
                resp = await orig(*args, **kwargs)
            except Exception as e:
                captured_exc = e
            t1 = time.time()
            try:
                req_body = (
                    kwargs if kwargs
                    else {"args": [str(a)[:200] for a in args]}
                )
                resp_body: Any = None
                if resp is not None:
                    md = getattr(resp, "model_dump", None)
                    resp_body = md() if callable(md) else str(resp)
                rec = {
                    "started_at": t0,
                    "ended_at": t1,
                    "duration_seconds": round(t1 - t0, 4),
                    "ok": captured_exc is None,
                    "error": repr(captured_exc) if captured_exc else None,
                    "request": req_body,
                    "response": resp_body,
                }
                with open(self._live_http_log_path, "a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
                    f.flush()
            except Exception:
                pass
            if captured_exc is not None:
                raise captured_exc
            return resp

        client.chat.completions.create = _logged_create  # type: ignore[assignment]
        try:
            setattr(client, "_swe_logging_wrapped", True)
        except Exception:
            pass


    @property
    def model_id(self) -> str:
        return self._model

    async def get_response(
        self, system_instructions, input, model_settings, tools,
        output_schema, handoffs, tracing, previous_response_id=None,
        conversation_id=None, prompt=None,
    ):
        if tools and self._quirk is not None:
            for t in tools:
                if getattr(t, "_swe_schema_simplified", False):
                    continue
                schema = getattr(t, "params_json_schema", None)
                if isinstance(schema, dict):
                    self._quirk(schema)
                try:
                    setattr(t, "_swe_schema_simplified", True)
                except Exception:
                    pass

        if model_settings is not None:
            if model_settings.temperature is None:
                model_settings.temperature = self._temperature
            if model_settings.top_p is None:
                model_settings.top_p = self._top_p
            if model_settings.presence_penalty is None:
                model_settings.presence_penalty = self._presence_penalty
            if (
                getattr(model_settings, "frequency_penalty", None) is None
                and self._frequency_penalty
            ):
                model_settings.frequency_penalty = self._frequency_penalty
            if (
                self._tool_choice
                and getattr(model_settings, "tool_choice", None) in (None, "auto")
            ):
                model_settings.tool_choice = self._tool_choice
            if self._seed is not None:
                if model_settings.extra_body is None:
                    model_settings.extra_body = {}
                model_settings.extra_body.setdefault("seed", self._seed)

        return await self._inner.get_response(
            system_instructions, input, model_settings, tools,
            output_schema, handoffs, tracing, previous_response_id,
            conversation_id, prompt,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
