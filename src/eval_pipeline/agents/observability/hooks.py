from __future__ import annotations

import contextvars
import json
import time
from typing import Any

from eval_pipeline.agents.observability.call_log import LLMCallLog
from eval_pipeline.agents.runtime.agent_registry import (
    clear_last_child_agent_id,
    last_child_agent_id,
)
from eval_pipeline.agents.runtime.turn import (
    ExecStats,
    LLMStats,
    Observation,
    Turn,
)
from eval_pipeline.agents.tools._state import _LAST_TOOL_EVENT_PAIR


SUCCESS_ATTEMPT_WINDOW: contextvars.ContextVar[dict | None] = (
    contextvars.ContextVar("SUCCESS_ATTEMPT_WINDOW", default=None)
)

try:
    from agents import RunHooks as _RunHooksBase
except ImportError:
    _RunHooksBase = object  # type: ignore[assignment,misc]


class LLMCallLogHooks(_RunHooksBase):

    def __init__(
        self,
        call_log: LLMCallLog,
        agent_id: str,
        agent_role: str,
        model: str = "",
        tokenizer: Any = None,
        block_size: int = 16,
        energy_url: str | None = None,
        idle_power_W: list[float] | None = None,
        metrics_url: str | None = None,
        agent_type: str | None = None,
        parent_agent_id: str | None = None,
    ) -> None:
        self.call_log = call_log
        self.agent_id = agent_id
        self.agent_role = agent_role
        # agent_type is the role/spec name; defaults to agent_role for
        # callers that haven't been migrated to pass it explicitly.
        self.agent_type = agent_type if agent_type is not None else agent_role
        self.parent_agent_id = parent_agent_id
        self.model = model
        self._start_time = 0.0
        self._current_input: list[dict] = []
        # The turn opened by the most recent on_llm_end; observations from
        # subsequent tool executions are appended to it synchronously.
        self._current_turn: Turn | None = None
        self._turn_index = 0
        self._tool_t0: float = 0.0
        self._tokenizer = tokenizer
        self._block_size = block_size
        self._prev_token_ids: list[int] = []
        self.energy_url = energy_url
        self._http: Any = None
        self._e_start: dict | None = None
        self._call_started_at: float = 0.0
        self.idle_power_W: list[float] = list(idle_power_W or [])
        # Auto-derived from ``energy_url`` if not given explicitly so
        if metrics_url is None and energy_url and energy_url.endswith("/energy"):
            metrics_url = energy_url[: -len("/energy")] + "/metrics"
        self.metrics_url = metrics_url
        self._cache_start: tuple[float, float] | None = None

    async def _snapshot_cache_metrics(self) -> tuple[float, float] | None:
        if not self.metrics_url:
            return None
        if self._http is None:
            try:
                import httpx
                self._http = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
            except Exception:
                return None
        try:
            r = await self._http.get(self.metrics_url)
            if r.status_code != 200:
                return None
            body = r.text
        except Exception:
            return None
        import re
        q = re.search(
            r'^vllm:prefix_cache_queries_total\{[^}]*\}\s+([0-9.eE+-]+)',
            body, re.M,
        )
        h = re.search(
            r'^vllm:prefix_cache_hits_total\{[^}]*\}\s+([0-9.eE+-]+)',
            body, re.M,
        )
        if not q or not h:
            return None
        try:
            return float(q.group(1)), float(h.group(1))
        except Exception:
            return None

    @staticmethod
    async def sample_idle_power(
        energy_url: str | None,
        samples: int = 3,
        interval_s: float = 0.5,
    ) -> list[float]:
        if not energy_url:
            return []
        try:
            import asyncio
            import httpx
        except Exception:
            return []
        accum: list[list[float]] = []
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(2.0),
            ) as hc:
                for _ in range(max(1, samples)):
                    r = await hc.get(energy_url)
                    if r.status_code != 200:
                        return []
                    data = r.json()
                    pw = data.get("gpus_power_W") or []
                    if not pw:
                        return []
                    accum.append([float(x) for x in pw])
                    await asyncio.sleep(interval_s)
        except Exception:
            return []
        if not accum:
            return []
        n_gpu = len(accum[0])
        return [
            sum(s[i] for s in accum) / len(accum)
            for i in range(n_gpu)
        ]


    async def _snapshot_energy(self) -> dict | None:
        if not self.energy_url:
            return None
        if self._http is None:
            try:
                import httpx
                self._http = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
            except Exception:
                return None
        for attempt in range(2):
            try:
                r = await self._http.get(self.energy_url)
                if r.status_code != 200:
                    if attempt == 0:
                        continue
                    return None
                return r.json()
            except Exception:
                if attempt == 0:
                    continue
                return None
        return None

    @staticmethod
    def _energy_diff_joules(
        e0: dict | None, e1: dict | None,
    ) -> tuple[float, list[float]]:
        if not e0 or not e1:
            return 0.0, []
        a = e0.get("gpus_mJ") or []
        b = e1.get("gpus_mJ") or []
        per_gpu_J: list[float] = []
        for x, y in zip(a, b):
            if x is None or y is None:
                per_gpu_J.append(0.0)
                continue
            per_gpu_J.append(max(0.0, (y - x) / 1000.0))
        return sum(per_gpu_J), per_gpu_J

    def _compute_cached_tokens(
        self, messages: list[Any],
    ) -> tuple[int, int]:
        if self._tokenizer is None:
            return 0, 0
        try:
            token_ids: list[int] = []
            for it in messages:
                segment = self._item_to_text_segment(it)
                if not segment:
                    continue
                ids = self._tokenizer.encode(segment, add_special_tokens=False)
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                token_ids.extend(ids)
            total = len(token_ids)
            common = 0
            for a, b in zip(self._prev_token_ids, token_ids):
                if a != b:
                    break
                common += 1
            cached = (common // self._block_size) * self._block_size
            self._prev_token_ids = token_ids
            return total, cached
        except Exception:
            return 0, 0

    @staticmethod
    def _item_to_text_segment(it: Any) -> str:
        if not isinstance(it, dict):
            return ""
        t = it.get("type")
        role = it.get("role")
        if t == "function_call":
            name = it.get("name") or ""
            args = it.get("arguments") or ""
            if not isinstance(args, str):
                args = str(args)
            return f"<|fc|>name={name}\nargs={args}\n<|/fc|>"
        if t == "function_call_output":
            out = it.get("output") or ""
            return f"<|fco|>{out}\n<|/fco|>"
        if t == "message" or role in ("system", "user", "assistant", "tool"):
            content = it.get("content", "")
            if isinstance(content, list):
                parts = []
                for b in content:
                    if isinstance(b, dict):
                        txt = b.get("text") or b.get("content") or ""
                        if txt:
                            parts.append(str(txt))
                content = "\n".join(parts)
            if not isinstance(content, str):
                content = str(content or "")
            return f"<|{role or 'assistant'}|>{content}\n<|/{role or 'assistant'}|>"
        return ""

    @staticmethod
    def _to_dict(item: Any) -> Any:
        if isinstance(item, dict):
            return item
        for attr in ("model_dump", "dict"):
            f = getattr(item, attr, None)
            if callable(f):
                try:
                    return f()
                except Exception:
                    pass
        try:
            return dict(item)
        except Exception:
            return {"__repr__": str(item)}

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        self._start_time = time.monotonic()
        self._call_started_at = time.time()
        self._e_start = None
        self._cache_start = await self._snapshot_cache_metrics()

        full_input: list[dict] = []
        if system_prompt:
            full_input.append({"role": "system", "content": str(system_prompt)})
        if isinstance(input_items, list):
            for it in input_items:
                full_input.append(self._to_dict(it))
        elif input_items:
            full_input.append({"role": "user", "content": str(input_items)})
        self._current_input = full_input

    async def on_llm_end(self, context, agent, response):
        window = SUCCESS_ATTEMPT_WINDOW.get()
        if window is not None:
            duration = window.get("duration") or 0.0
            ended_at = window.get("ended_at") or time.time()
            e_end = window.get("e_end")
            e_start = window.get("e_start")
            n_attempts = window.get("n_attempts", 1)
            started_at_use = window.get("started_at") or self._call_started_at
            SUCCESS_ATTEMPT_WINDOW.set(None)
        else:
            duration = time.monotonic() - self._start_time
            ended_at = time.time()
            e_end = await self._snapshot_energy()
            e_start = self._e_start
            n_attempts = 1
            started_at_use = self._call_started_at
        if e_start and e_end:
            j_total_raw, j_per_gpu_raw = self._energy_diff_joules(
                e_start, e_end,
            )
            idle_J_total = sum(self.idle_power_W) * duration
            j_total: float | None = max(0.0, j_total_raw - idle_J_total)
            j_per_gpu: list[float] = []
            for i, v in enumerate(j_per_gpu_raw):
                idle_W_i = (
                    self.idle_power_W[i]
                    if i < len(self.idle_power_W)
                    else 0.0
                )
                j_per_gpu.append(max(0.0, v - idle_W_i * duration))
            joules_val = j_total
            joules_raw_val = j_total_raw
        else:
            j_per_gpu = []
            j_per_gpu_raw = []
            joules_val = None
            joules_raw_val = None

        input_tokens = output_tokens = cached_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = response.usage.input_tokens or 0
            output_tokens = response.usage.output_tokens or 0
            if (
                hasattr(response.usage, "input_tokens_details")
                and response.usage.input_tokens_details
            ):
                cached_tokens = getattr(
                    response.usage.input_tokens_details, "cached_tokens", 0,
                ) or 0

        if cached_tokens == 0 and self._tokenizer is not None:
            try:
                _, cached_tokens = self._compute_cached_tokens(self._current_input)
            except Exception:
                cached_tokens = 0
            if input_tokens > 0:
                # The client-side estimate can overshoot the server's count
                # when tokenizers disagree (e.g. tekken); cached tokens are a
                # subset of the prompt, so clamp to the reported input.
                cached_tokens = min(cached_tokens, input_tokens)

        cache_q_delta: float | None = None
        cache_h_delta: float | None = None
        cache_hit_ratio: float | None = None
        if self._cache_start is not None:
            cache_end = await self._snapshot_cache_metrics()
            if cache_end is not None:
                dq = max(0.0, cache_end[0] - self._cache_start[0])
                dh = max(0.0, cache_end[1] - self._cache_start[1])
                cache_q_delta = dq
                cache_h_delta = dh
                cache_hit_ratio = (dh / dq) if dq > 0 else None
        self._cache_start = None

        raw_output: list[dict] = []
        action: list[dict] = []
        text_parts: list[str] = []
        if hasattr(response, "output"):
            for item in response.output:
                d = self._to_dict(item)
                raw_output.append(d)
                if d.get("type") == "function_call" or d.get("name"):
                    action.append(d)
                else:
                    c = d.get("content")
                    if isinstance(c, str):
                        text_parts.append(c)
                    elif isinstance(c, list):
                        for blk in c:
                            if isinstance(blk, dict):
                                if blk.get("text"):
                                    text_parts.append(blk["text"])
                                elif blk.get("content"):
                                    text_parts.append(str(blk["content"]))

        turn = Turn(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            turn_index=self._turn_index,
            parent_agent_id=self.parent_agent_id,
            llm_input=self._current_input,
            llm_output_text="".join(text_parts),
            raw_output=raw_output,
            parsed_actions=action,
            llm_stats=LLMStats(
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                duration_seconds=round(duration, 3),
                n_attempts=n_attempts,
                joules=joules_val,
                joules_per_gpu=j_per_gpu,
                joules_raw=joules_raw_val,
                joules_per_gpu_raw=j_per_gpu_raw,
                idle_power_W=list(self.idle_power_W),
                cache_q_delta=cache_q_delta,
                cache_h_delta=cache_h_delta,
                cache_hit_ratio=cache_hit_ratio,
                started_at=started_at_use,
                ended_at=ended_at,
            ),
            observations=[],
        )
        self.call_log.record_turn(turn)
        self._current_turn = turn
        self._turn_index += 1

    async def on_tool_start(self, context, agent, tool):
        self._tool_t0 = time.monotonic()
        # So a non agent-call tool doesn't inherit a stale child link; an
        # agent-call tool re-sets it via invoke_agent during execution.
        clear_last_child_agent_id()

    async def on_tool_end(self, context, agent, tool, result):
        """Bind the execution result to the open turn *synchronously* —
        one Observation per tool call, with per-execution statistics
        recovered from the typed ``_LAST_TOOL_EVENT_PAIR``."""
        tool_name = getattr(tool, "name", str(tool))
        result_str = str(result)
        duration_s = max(0.0, time.monotonic() - self._tool_t0)

        # Recover exit_code / full (untruncated) content / args from the
        # typed action+observation pair the tool wrapper stashed.
        exit_code: int | None = None
        full_content: str | None = None
        args: dict = {}
        pair = _LAST_TOOL_EVENT_PAIR.get()
        if pair is not None:
            action_obj, obs_obj = pair
            exit_code = getattr(obs_obj, "exit_code", None)
            c = getattr(obs_obj, "content", None)
            full_content = c if isinstance(c, str) else None
            args = _extract_tool_args(action_obj)

        payload = full_content if full_content is not None else result_str
        truncated = (
            full_content is not None and len(full_content) > len(result_str)
        )

        obs = Observation(
            tool_name=tool_name,
            args=args,
            payload=payload,
            exec_stats=ExecStats(
                exit_code=exit_code,
                duration_seconds=round(duration_s, 3),
                output_len=len(payload),
                truncated=truncated,
            ),
            child_agent_id=last_child_agent_id(),
        )
        if self._current_turn is not None:
            self._current_turn.observations.append(obs)
            self.call_log.update_live()

    async def on_agent_start(self, context, agent):
        pass

    async def on_handoff(self, context, from_agent, to_agent):
        pass

    async def on_agent_end(self, context, agent, output):
        # Observations are bound synchronously in on_tool_end now; nothing
        # to flush here.
        pass


def _extract_tool_args(action_obj: Any) -> dict:
    """Recover the tool arguments from a typed action object.

    ``CmdRunAction`` carries ``.command``; ``ToolCallAction`` carries a
    JSON ``.args_json``. Anything else yields an empty dict.
    """
    if action_obj is None:
        return {}
    cmd = getattr(action_obj, "command", None)
    if cmd is not None:
        return {"command": cmd}
    args_json = getattr(action_obj, "args_json", None)
    if isinstance(args_json, str):
        try:
            return json.loads(args_json)
        except Exception:
            return {"args_json": args_json}
    return {}


async def setup_energy_telemetry(
    base_url: str | None,
    *,
    label: str = "Energy",
) -> tuple[str | None, list[float]]:
    energy_url = (
        base_url.rstrip("/").removesuffix("/v1") + "/energy"
        if base_url else None
    )
    idle_power_W = await LLMCallLogHooks.sample_idle_power(energy_url)
    if idle_power_W:
        print(
            f"[{label}] idle baseline (per-GPU W): "
            f"{[round(p, 1) for p in idle_power_W]}"
        )
    return energy_url, idle_power_W
