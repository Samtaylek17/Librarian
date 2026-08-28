from __future__ import annotations

from typing import Any, Protocol


class SchemaQuirk(Protocol):

    def __call__(self, schema: Any) -> Any: ...


def _simplify_tool_schema_for_qwen(schema: Any) -> Any:
    if isinstance(schema, dict):
        any_of = schema.get("anyOf")
        if isinstance(any_of, list) and len(any_of) == 2:
            non_null = [
                b for b in any_of
                if not (isinstance(b, dict) and b.get("type") == "null")
            ]
            null_count = len(any_of) - len(non_null)
            if (null_count == 1 and len(non_null) == 1
                    and isinstance(non_null[0], dict)):
                kept = {k: v for k, v in schema.items() if k != "anyOf"}
                kept.update(non_null[0])
                schema.clear()
                schema.update(kept)
        for v in schema.values():
            _simplify_tool_schema_for_qwen(v)
    elif isinstance(schema, list):
        for v in schema:
            _simplify_tool_schema_for_qwen(v)
    return schema


class QwenAnyOfNullQuirk:

    def __call__(self, schema: Any) -> Any:
        return _simplify_tool_schema_for_qwen(schema)
