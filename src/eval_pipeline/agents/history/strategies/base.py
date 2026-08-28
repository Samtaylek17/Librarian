from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class CompactionContext:

    def __init__(
        self,
        *,
        new_chars: int = 0,
        request: str = "",
        **extras: Any,
    ) -> None:
        self.new_chars = new_chars
        self.request = request
        for k, v in extras.items():
            setattr(self, k, v)


@runtime_checkable
class MemoryStrategy(Protocol):

    async def render(self, history: list, request: str) -> list:
        ...

    async def compact(
        self,
        history: list,
        session_delta: list,
        ctx: CompactionContext,
    ) -> list:
        ...


CompactionStrategy = MemoryStrategy


class _DefaultRenderMixin:

    async def render(self, history: list, request: str) -> list:  # noqa: ARG002
        return list(history)
