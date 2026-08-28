"""Per-episode LLM/runtime bootstrap shared by every MAS runner.

Every ``run_with_executor`` used to open with ~50 identical lines: build the
call log, load the tokenizer, open the OpenAI client, probe energy
telemetry, construct ``_ConfiguredModel``, and create the ``AgentRegistry``.
:func:`init_mas_runtime` does all of that once and returns a
:class:`MASRuntime`; it also assigns ``runner.call_log`` and
``runner._registry`` (read by :class:`AgentRunner.run` and tests).

:meth:`MASRuntime.logged` fills the runtime-derived keyword arguments of
``invoke_agent_logged`` so call sites only pass what is unique to them
(``agent_type`` / ``agent_obj_factory`` / ``execute`` / ``agent_id`` /
``mode``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from eval_pipeline.agents.llm import _ConfiguredModel
from eval_pipeline.agents.observability import LLMCallLog, setup_energy_telemetry
from eval_pipeline.agents.runtime import AgentRegistry

logger = logging.getLogger(__name__)


def _live_path(runner: Any, name: str) -> str | None:
    base = getattr(runner, "_live_log_dir", None)
    return f"{base}/{name}" if base else None


def _load_tokenizer(model: str, *, block_size_from_model: bool) -> tuple[Any, int]:
    """Return ``(tokenizer, block_size)``; ``(None, 16)`` if load fails.

    ``block_size`` only affects the cached-token estimate in the call log
    (prefix rounded down to a block boundary). Standalone librarian / vanilla
    historically derived it from the model; the MAS overlays hard-coded 16.
    ``block_size_from_model`` preserves that per-runner behaviour exactly.
    """
    try:
        from transformers import AutoTokenizer

        from eval_pipeline.utils.llm_proxy import LLMLoggingProxy

        tok_name = model.removeprefix("openai/")
        tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
        block_size = LLMLoggingProxy._get_block_size(tok_name) if block_size_from_model else 16
        return tokenizer, block_size
    except Exception as e:  # noqa: BLE001
        logger.debug("Tokenizer load failed: %s", e)
        return None, 16


@dataclass
class MASRuntime:
    """Episode-scoped LLM plumbing handed to a runner's wiring code."""

    model: str
    call_log: LLMCallLog
    tokenizer: Any
    block_size: int
    llm_model: _ConfiguredModel
    energy_url: str | None
    idle_power_W: list[float]
    registry: AgentRegistry
    # Per-role count of CONSECUTIVE stuck-terminates across delegations
    # (reset when that role returns cleanly). An orchestrator that keeps
    # re-delegating the same wedged role escalates to an episode-level
    # StuckTerminate once a role hits the budget — see _delegation.py.
    consec_stuck: dict[str, int] = field(default_factory=dict)

    def logged(self, **overrides: Any) -> dict[str, Any]:
        """Runtime-derived kwargs for ``invoke_agent_logged`` (+ overrides)."""
        kw = dict(
            registry=self.registry,
            call_log=self.call_log,
            model=self.model,
            tokenizer=self.tokenizer,
            block_size=self.block_size,
            energy_url=self.energy_url,
            idle_power_W=self.idle_power_W,
        )
        kw.update(overrides)
        return kw


async def init_mas_runtime(runner: Any, *, label: str) -> MASRuntime:
    """Build the per-episode :class:`MASRuntime` for ``runner``.

    Side effects: sets ``runner.call_log`` and ``runner._registry`` so the
    base ``AgentRunner.run`` can harvest turns and tests can introspect.
    """
    from openai import AsyncOpenAI

    call_log = LLMCallLog(live_dump_path=_live_path(runner, "live_calls.jsonl"))
    tokenizer, block_size = _load_tokenizer(
        runner.model,
        block_size_from_model=getattr(runner, "BLOCK_SIZE_FROM_MODEL", False),
    )

    api_key = runner.api_key or os.environ.get("OPENAI_API_KEY", "dummy")
    client = AsyncOpenAI(
        base_url=runner.base_url or None,
        api_key=api_key,
        timeout=300.0,
        max_retries=0,
    )
    energy_url, idle_power_W = await setup_energy_telemetry(
        runner.base_url, label=label,
    )
    llm_model = _ConfiguredModel(
        model=runner.model,
        openai_client=client,
        seed=runner.seed,
        tokenizer=tokenizer,
        live_http_log_path=_live_path(runner, "live_http.jsonl"),
        **runner.sampling_kwargs,
    )

    registry = AgentRegistry()
    runner.call_log = call_log
    runner._registry = registry
    return MASRuntime(
        model=runner.model,
        call_log=call_log,
        tokenizer=tokenizer,
        block_size=block_size,
        llm_model=llm_model,
        energy_url=energy_url,
        idle_power_W=idle_power_W,
        registry=registry,
    )
