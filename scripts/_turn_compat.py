"""Back-compat shims for reading trajectory entries.

Trajectory entries in episode JSON used to be flat per-LLM-call records;
they are now :class:`Turn` dicts (identity + nested ``llm_stats`` + a list
of ``observations``). These helpers let log-reading scripts handle both
shapes. Stdlib-only — safe to import from the lightweight base python.
"""

from __future__ import annotations

from typing import Any


def stat(call: dict, key: str, default: Any = None) -> Any:
    """Read a per-call statistic from either the nested ``llm_stats`` (new
    Turn dicts) or the flat top level (legacy logs)."""
    stats = call.get("llm_stats")
    if isinstance(stats, dict) and key in stats:
        return stats.get(key, default)
    return call.get(key, default)


def _observations_to_legacy(observations: list) -> list[dict]:
    """Render a Turn's observations list back to the legacy
    ``[{"tool_name", "output"}, ...]`` shape."""
    out = []
    for o in observations:
        if isinstance(o, dict):
            out.append({
                "tool_name": o.get("tool_name", ""),
                "output": o.get("payload", ""),
            })
    return out


def normalize_call(call: dict) -> dict:
    """Return a dict exposing the legacy flat field names regardless of
    whether ``call`` is a Turn dict or an old flat record.

    New Turn fields are mapped onto their legacy names (``llm_input`` →
    ``input``, ``parsed_actions`` → ``action``, ``observations`` →
    ``observation``, ``llm_output_text`` → ``text``, ``agent_type`` →
    ``agent_role``) and every ``llm_stats`` measurement is flattened to the
    top level. Already-flat records pass through unchanged.
    """
    if "llm_stats" not in call and "parsed_actions" not in call:
        return call  # legacy flat record

    out = dict(call)
    stats = call.get("llm_stats")
    if isinstance(stats, dict):
        for k, v in stats.items():
            out.setdefault(k, v)
    out["input"] = call.get("llm_input", call.get("input"))
    out["text"] = call.get("llm_output_text", call.get("text", ""))
    out["action"] = call.get("parsed_actions", call.get("action", []))
    out["agent_role"] = call.get("agent_type", call.get("agent_role", ""))
    obs = call.get("observations")
    if obs is not None:
        out["observation"] = _observations_to_legacy(obs)
    return out
