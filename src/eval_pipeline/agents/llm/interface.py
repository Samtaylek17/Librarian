from __future__ import annotations

from typing import Any, Protocol


class LLMInterface(Protocol):

    async def get_response(
        self,
        system_instructions: Any,
        input: Any,
        model_settings: Any,
        tools: Any,
        output_schema: Any,
        handoffs: Any,
        tracing: Any,
        previous_response_id: Any = None,
        conversation_id: Any = None,
        prompt: Any = None,
    ) -> Any: ...
