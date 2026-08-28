from __future__ import annotations

import contextvars
import json


# to build typed history entries (matching upstream OH stuck.py
_LAST_TOOL_EVENT_PAIR: contextvars.ContextVar = contextvars.ContextVar(
    "last_tool_event_pair", default=None,
)


def _canonical_args(d: dict) -> str:
    try:
        clean = {k: v for k, v in d.items() if v is not None}
        return json.dumps(clean, sort_keys=True, default=str)
    except Exception:
        return repr(d)
