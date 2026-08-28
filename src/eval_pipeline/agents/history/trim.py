from __future__ import annotations


def _trim_trailing_submit(history: list) -> list:
    if not history:
        return history

    cleaned = list(history)

    while cleaned:
        last = cleaned[-1]
        if not isinstance(last, dict):
            break
        t = last.get("type", "")

        if t == "function_call_output" and len(cleaned) >= 2:
            cid = last.get("call_id", "")
            matching_call = None
            for prior in reversed(cleaned[:-1]):
                if (isinstance(prior, dict)
                        and prior.get("type") == "function_call"
                        and prior.get("call_id") == cid):
                    matching_call = prior
                    break
            if (matching_call
                    and "submit" in matching_call.get("name", "").lower()):
                cleaned.pop()
                cleaned.remove(matching_call)
                continue

        if t == "function_call" and "submit" in last.get("name", "").lower():
            cleaned.pop()
            continue

        break

    return cleaned
