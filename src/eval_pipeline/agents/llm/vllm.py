from __future__ import annotations

import json
import os
import time
from typing import Any

from eval_pipeline.agents.llm.quirks import (
    SchemaQuirk,
    _simplify_tool_schema_for_qwen,
)


def _is_transient_error(e: Exception) -> bool:
    name = type(e).__name__
    msg = str(e).lower()
    if name in (
        "ReadTimeout", "ConnectTimeout", "WriteTimeout", "PoolTimeout",
        "TimeoutException", "ReadError", "WriteError", "ConnectError",
        "RemoteProtocolError", "ProxyError", "NetworkError",
        "ConnectionError",
    ):
        return True
    if name in (
        "APIConnectionError", "APITimeoutError", "InternalServerError",
    ):
        return True
    if any(s in msg for s in ("503 service unavailable", "504 gateway timeout",
                              "502 bad gateway", "500 internal server")):
        return True
    if name == "RateLimitError":
        return True
    return False


def _parse_energy_headers(headers) -> tuple[dict | None, dict | None]:
    def _get(name: str) -> str | None:
        try:
            v = headers.get(name)
        except Exception:
            v = None
        if v is None:
            try:
                v = headers.get(name.lower())
            except Exception:
                v = None
        return v
    ts_s = _get("X-Energy-T-Start")
    te_s = _get("X-Energy-T-End")
    ms_s = _get("X-Energy-MJ-Start")
    me_s = _get("X-Energy-MJ-End")
    if not (ts_s and te_s and ms_s is not None and me_s is not None):
        return (None, None)
    try:
        ts = float(ts_s)
        te = float(te_s)
    except Exception:
        return (None, None)

    def _parse_csv(s: str) -> list:
        out: list = []
        for x in s.split(","):
            x = x.strip()
            if not x:
                out.append(None)
            else:
                try:
                    out.append(int(x) if x.isdigit() else float(x))
                except Exception:
                    out.append(None)
        return out

    gpus_s = _parse_csv(ms_s)
    gpus_e = _parse_csv(me_s)
    return (
        {"t": ts, "gpus_mJ": gpus_s, "n_gpus": len(gpus_s)},
        {"t": te, "gpus_mJ": gpus_e, "n_gpus": len(gpus_e)},
    )


def _is_tool_call_parse_error(e: Exception) -> bool:
    name = type(e).__name__
    if name != "BadRequestError":
        return False
    msg = str(e).lower()
    if "maximum context length" in msg or "context_length_exceeded" in msg:
        return False
    return (
        "unterminated string" in msg
        or ("tool" in msg and "parse" in msg)
    )


class VllmLLM:
    """vLLM-served LLM. Sampling + seed + /energy + raw HTTP log.

    All vLLM-specific knobs live here; the agent code doesn't see them.
    Implements the SDK's ``Model`` duck-type via :meth:`get_response`.

    Seed wiring: forwarded to vLLM via ``extra_body['seed']`` so two
    episodes with the same seed sample identically (modulo vLLM's
    floating-point reduction order under different batch compositions).

    No memory condensation — input is passed through verbatim. When
    the rendered request exceeds the context window, vLLM returns a
    400 BadRequestError; ``safe_runner_run`` translates that into
    ``ContextWindowExceeded`` (graceful terminal).
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        openai_client: Any = None,
        seed: int | None = None,
        tokenizer: Any = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        tool_choice: str | None = None,
        live_http_log_path: str | None = None,
        quirk: SchemaQuirk | None = None,
        request_timeout: float | None = None,
        # to ``None`` for unbounded retry until success (the upstream
        max_manual_retries: int | None = None,
        thinking_on: bool = True,
    ):
        from agents.models.interface import Model
        from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

        if openai_client is None:
            if not base_url:
                raise ValueError(
                    "VllmLLM: pass either ``base_url`` or a pre-built "
                    "``openai_client``."
                )
            from openai import AsyncOpenAI
            openai_client = AsyncOpenAI(
                base_url=base_url,
                api_key="dummy",
                timeout=request_timeout,
                max_retries=0,
            )
        else:
            wo = getattr(openai_client, "with_options", None)
            if callable(wo):
                try:
                    openai_client = wo(max_retries=0)
                except Exception:
                    pass

        self._inner = OpenAIChatCompletionsModel(
            model=model, openai_client=openai_client,
        )
        self._max_manual_retries = max_manual_retries
        self._seed = seed
        self._openai_client = openai_client
        self._model = model
        self._tokenizer = tokenizer
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._min_p = min_p
        self._presence_penalty = presence_penalty
        self._repetition_penalty = repetition_penalty
        self._tool_choice = tool_choice
        self._thinking_on = thinking_on
        self._quirk: SchemaQuirk = (
            quirk if quirk is not None else _simplify_tool_schema_for_qwen
        )
        Model.register(VllmLLM)
        self._live_http_log_path = live_http_log_path
        if live_http_log_path:
            try:
                os.makedirs(os.path.dirname(live_http_log_path), exist_ok=True)
                open(live_http_log_path, "w").close()
            except Exception:
                pass
        self._wrap_chat_completions_for_logging()


    def _wrap_chat_completions_for_logging(self) -> None:
        client = self._openai_client
        if getattr(client, "_swe_logging_wrapped", False):
            return
        orig = client.chat.completions.create
        orig_raw = client.chat.completions.with_raw_response.create
        log_path = self._live_http_log_path
        max_retries = self._max_manual_retries

        from eval_pipeline.agents.observability.hooks import (
            SUCCESS_ATTEMPT_WINDOW,
        )

        async def _logged_create(*args, **kwargs):
            import asyncio

            from eval_pipeline.agents.runtime.agent_registry import (
                enforce_llm_agent_gate,
            )

            # Requirement 1: only agents may call the LLM. No-op unless
            # EVAL_LLM_AGENT_GATE is set to warn/strict.
            enforce_llm_agent_gate()

            SUCCESS_ATTEMPT_WINDOW.set(None)

            attempts: list[dict] = []
            captured_exc: Exception | None = None
            resp = None
            attempt_idx = 0
            t_chain_start = time.time()
            req_body_for_log: dict = (
                kwargs if kwargs
                else {"args": [str(a)[:200] for a in args]}
            )

            orig_kwargs = kwargs
            orig_seed = None
            if isinstance(orig_kwargs.get("extra_body"), dict):
                orig_seed = orig_kwargs["extra_body"].get("seed")

            while True:
                attempt_idx += 1
                t0 = time.time()
                attempt_exc: Exception | None = None
                e_start_hdr: dict | None = None
                e_end_hdr: dict | None = None
                try:
                    raw = await orig_raw(*args, **kwargs)
                    resp = raw.parse()
                    if hasattr(resp, "__await__"):
                        resp = await resp
                    e_start_hdr, e_end_hdr = _parse_energy_headers(raw.headers)
                except Exception as e:
                    attempt_exc = e
                t1 = time.time()
                attempts.append({
                    "attempt": attempt_idx,
                    "started_at": t0,
                    "ended_at": t1,
                    "duration_seconds": round(t1 - t0, 4),
                    "ok": attempt_exc is None,
                    "error": repr(attempt_exc) if attempt_exc else None,
                    "energy_headers_present": (
                        e_start_hdr is not None and e_end_hdr is not None
                    ),
                })
                if attempt_exc is None:
                    captured_exc = None
                    # ``"}`` truncated. The SDK stores this verbatim
                    self._repair_broken_tool_call_args(resp)
                    if e_start_hdr is not None and e_end_hdr is not None:
                        win_start = e_start_hdr["t"]
                        win_end = e_end_hdr["t"]
                    else:
                        win_start = t0
                        win_end = t1
                    SUCCESS_ATTEMPT_WINDOW.set({
                        "started_at": win_start,
                        "ended_at": win_end,
                        "duration": win_end - win_start,
                        "e_start": e_start_hdr,
                        "e_end": e_end_hdr,
                        "n_attempts": attempt_idx,
                    })
                    break

                is_parse_err = _is_tool_call_parse_error(attempt_exc)
                is_transient = _is_transient_error(attempt_exc)
                if not is_parse_err and not is_transient:
                    captured_exc = attempt_exc
                    break
                if max_retries is not None and attempt_idx > max_retries:
                    captured_exc = attempt_exc
                    break

                if is_parse_err:
                    new_kwargs = dict(kwargs)
                    eb = dict(new_kwargs.get("extra_body") or {})
                    eb["seed"] = (
                        (orig_seed if orig_seed is not None else 0)
                        + attempt_idx
                    )
                    new_kwargs["extra_body"] = eb
                    kwargs = new_kwargs
                    continue

                backoff = min(30.0, 2.0 ** (attempt_idx - 1))
                await asyncio.sleep(backoff)

            t_chain_end = time.time()
            if log_path:
                try:
                    resp_body: Any = None
                    if resp is not None:
                        md = getattr(resp, "model_dump", None)
                        resp_body = md() if callable(md) else str(resp)
                    raw_prompt = await self._fetch_rendered_prompt(
                        req_body_for_log,
                    )
                    raw_completion = self._reconstruct_raw_output(resp_body)
                    reasoning = self._extract_reasoning(resp_body)
                    rec = {
                        "started_at": t_chain_start,
                        "ended_at": t_chain_end,
                        "duration_seconds": round(
                            t_chain_end - t_chain_start, 4,
                        ),
                        "ok": captured_exc is None,
                        "error": repr(captured_exc) if captured_exc else None,
                        "n_attempts": attempt_idx,
                        "attempts": attempts,
                        "request": req_body_for_log,
                        "response": resp_body,
                        "raw_prompt": raw_prompt,
                        "raw_completion": raw_completion,
                        "reasoning": reasoning,
                    }
                    with open(log_path, "a") as f:
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

    async def _fetch_rendered_prompt(self, req_body: dict) -> str | None:
        base = getattr(self._openai_client, "base_url", None)
        if base is None:
            return None
        try:
            base_str = str(base).rstrip("/").removesuffix("/v1")
        except Exception:
            return None
        if not base_str:
            return None
        try:
            import httpx
            tok_body: dict = {
                "model": self._model,
                "messages": req_body.get("messages") or [],
                "add_generation_prompt": True,
            }
            if req_body.get("tools"):
                tok_body["tools"] = req_body["tools"]
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as hc:
                rt = await hc.post(f"{base_str}/tokenize", json=tok_body)
                if rt.status_code != 200:
                    return None
                ids = rt.json().get("tokens")
                if not ids:
                    return None
                rd = await hc.post(
                    f"{base_str}/detokenize",
                    json={"model": self._model, "tokens": ids},
                )
                if rd.status_code != 200:
                    return None
                return rd.json().get("prompt")
        except Exception:
            return None

    @staticmethod
    def _try_close_truncated_json(s: str) -> str | None:
        for closer in ('"}', '"]}', '"]"}', '}', ']}', '"}]'):
            try:
                json.loads(s + closer)
                return s + closer
            except Exception:
                pass
        return None

    @classmethod
    def _repair_broken_tool_call_args(cls, resp) -> None:
        try:
            choices = getattr(resp, "choices", None) or []
            for ch in choices:
                msg = getattr(ch, "message", None)
                if msg is None:
                    continue
                tcs = getattr(msg, "tool_calls", None) or []
                for tc in tcs:
                    fn = getattr(tc, "function", None)
                    if fn is None:
                        continue
                    args = getattr(fn, "arguments", None)
                    if not isinstance(args, str):
                        continue
                    try:
                        json.loads(args)
                        continue
                    except Exception:
                        pass
                    repaired = cls._try_close_truncated_json(args)
                    if repaired is None:
                        repaired = json.dumps({
                            "_args_unparseable_raw": args,
                        })
                    fn.arguments = repaired
        except Exception:
            pass

    @staticmethod
    def _reconstruct_raw_output(resp_body: dict | None) -> str | None:
        """Stitch what vLLM returned into a single human-readable string.

        Intentionally NO parsing — ``arguments`` is dumped verbatim as
        the JSON string vLLM gave us. We do not call ``json.loads`` and
        we do not re-render into ``<parameter=K>V</parameter>`` form,
        even though that's the shape the (qwen3_xml / seed_oss) model
        originally emitted. Reason: parsing-then-reconstructing hides
        exactly the cases we need to debug — when vLLM's tool parser
        emits truncated / malformed JSON, the parse silently fails and
        the reconstruction drops the parameter content entirely. By
        recording vLLM's bytes verbatim, log_viewer shows what the
        server actually returned.
        """
        if not resp_body:
            return None
        try:
            ch = (resp_body.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            parts: list[str] = []
            if msg.get("content"):
                parts.append(msg["content"])
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                args_raw = fn.get("arguments")
                if args_raw is None:
                    args_str = ""
                elif isinstance(args_raw, str):
                    args_str = args_raw
                else:
                    try:
                        args_str = json.dumps(args_raw)
                    except Exception:
                        args_str = str(args_raw)
                parts.append(
                    f"<tool_call>\n<function={name}>\n{args_str}\n"
                    f"</function>\n</tool_call>"
                )
            return "\n".join(parts) if parts else None
        except Exception:
            return None

    @staticmethod
    def _extract_reasoning(resp_body: dict | None) -> str | None:
        """Pull the model's chain-of-thought from vLLM's response.

        When a ``--reasoning-parser`` is active, vLLM splits the thinking
        block out of ``content`` into ``message.reasoning`` (the OpenAI SDK
        keeps it as a ``model_extra`` field, so it survives ``model_dump``).
        ``_reconstruct_raw_output`` deliberately omits it; we capture it
        separately so the trajectory log retains the reasoning without
        polluting the verbatim output stitch. Falls back to the older
        ``reasoning_content`` key for forward/backward compatibility.
        """
        if not resp_body:
            return None
        try:
            ch = (resp_body.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            r = msg.get("reasoning") or msg.get("reasoning_content")
            return r if r else None
        except Exception:
            return None

    @property
    def model_id(self):
        return self._inner.model if hasattr(self._inner, 'model') else "unknown"

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
            if model_settings.extra_body is None:
                model_settings.extra_body = {}
            model_settings.extra_body.setdefault("top_k", self._top_k)
            model_settings.extra_body.setdefault("min_p", self._min_p)
            model_settings.extra_body.setdefault(
                "repetition_penalty", self._repetition_penalty,
            )
            _ = self._seed
            if self._tool_choice and getattr(model_settings, "tool_choice", None) in (None, "auto"):
                model_settings.tool_choice = self._tool_choice
            from eval_pipeline.agents.llm.profiles import thinking_extra_body
            tek = thinking_extra_body(self._model, on=self._thinking_on)
            tek_ctk = tek.get("chat_template_kwargs", {})
            if tek_ctk:
                # Only create the key when the profile has kwargs to inject:
                # vLLM's Mistral-tokenizer path 400s on the mere presence of
                # chat_template_kwargs, even an empty dict.
                ctk = model_settings.extra_body.setdefault("chat_template_kwargs", {})
                for k, v in tek_ctk.items():
                    ctk.setdefault(k, v)
        return await self._inner.get_response(
            system_instructions, input, model_settings, tools,
            output_schema, handoffs, tracing, previous_response_id,
            conversation_id, prompt,
        )

    def __getattr__(self, name):
        return getattr(self._inner, name)
