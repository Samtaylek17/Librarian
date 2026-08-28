from __future__ import annotations


def wrap_continuation_request(
    request: str, caller_role: str, freshness: str
) -> str:
    return (
        f"<tool_response>\n"
        f"[CONTINUATION REQUEST from {caller_role}]:\n"
        f"{request}\n\n"
        f"{freshness}\n"
        f"</tool_response>"
    )
