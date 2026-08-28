"""SWE-Agent style ``last-N-observations`` context overlay.

Keeps the last ``n`` ``function_call_output`` items (tool observations)
verbatim and replaces the ``output`` body of older ones with a fixed
marker so prompts stay short across long agent loops.

Items NOT touched: system / user / message / function_call. Only the
``output`` STRING of older ``function_call_output`` items is rewritten —
``call_id``, ``type``, and surrounding ``function_call`` actions are
preserved so the call/output linkage and the action trace remain intact.

Origin: SWE-agent ``history_processors.LastNObservations`` (sweagent/
agent/history_processors.py). Default ``n=5`` matches upstream.

Cache note: the marker text is FIXED (no original length, no per-call
content) so two elided observations are byte-identical and vLLM's
prefix cache hits across them.
"""

from __future__ import annotations

from copy import copy
from typing import Any, Mapping


ELISION_MARKER = "[Observation elided — last-N policy]"


def _register_with_agents_sdk() -> None:
    try:
        from agents.models.interface import Model as _SdkModel
        _SdkModel.register(LastNObservationModel)
    except Exception:
        pass


def elide_old_observations(
    input_items: list[Any],
    *,
    last_n: int,
    marker: str = ELISION_MARKER,
) -> list[Any]:
    if last_n is None:
        raise ValueError("last_n must be an int, not None")
    if last_n < 0:
        raise ValueError(f"last_n must be >= 0, got {last_n}")

    obs_indices = [
        i for i, it in enumerate(input_items)
        if isinstance(it, dict) and it.get("type") == "function_call_output"
    ]
    if len(obs_indices) <= last_n:
        return list(input_items)

    cutoff = set(obs_indices[: len(obs_indices) - last_n])

    out: list[Any] = []
    for i, it in enumerate(input_items):
        if i in cutoff:
            new = copy(it)
            new["output"] = marker
            out.append(new)
        else:
            out.append(it)
    return out


class LastNObservationModel:

    def __init__(self, inner: Any, *, last_n: int, marker: str = ELISION_MARKER) -> None:
        if last_n is None:
            raise ValueError("last_n must be an int, not None")
        if last_n < 0:
            raise ValueError(f"last_n must be >= 0, got {last_n}")
        self._inner = inner
        self._last_n = last_n
        self._marker = marker

    @property
    def model_id(self) -> str:
        return getattr(self._inner, "model_id", "unknown")

    async def get_response(
        self, system_instructions, input, model_settings, tools,
        output_schema, handoffs, tracing, previous_response_id=None,
        conversation_id=None, prompt=None,
    ):
        elided = elide_old_observations(
            input, last_n=self._last_n, marker=self._marker,
        )
        return await self._inner.get_response(
            system_instructions, elided, model_settings, tools,
            output_schema, handoffs, tracing, previous_response_id,
            conversation_id, prompt,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


_register_with_agents_sdk()


def resolve_per_role(
    spec: int | Mapping[str, int] | None,
    role: str,
) -> int | None:
    if spec is None:
        return None
    if isinstance(spec, int):
        return spec
    if isinstance(spec, Mapping):
        v = spec.get(role)
        if v is None:
            return None
        if not isinstance(v, int):
            raise TypeError(
                f"last_n_observations[{role!r}] must be int, got {type(v).__name__}"
            )
        return v
    raise TypeError(
        f"last_n_observations must be int | Mapping[str,int] | None, "
        f"got {type(spec).__name__}"
    )


def maybe_wrap_last_n(
    inner: Any,
    spec: int | Mapping[str, int] | None,
    role: str,
) -> Any:
    n = resolve_per_role(spec, role)
    if n is None:
        return inner
    return LastNObservationModel(inner, last_n=n)
