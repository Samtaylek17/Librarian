from __future__ import annotations

from typing import Any

try:
    from agents import RunHooks as _RunHooksBase
except ImportError:
    _RunHooksBase = object  # type: ignore[assignment,misc]


class _MultiHooks(_RunHooksBase):

    def __init__(self, children: list[Any]) -> None:
        self.children = children

    async def _fan(self, name: str, *args, **kwargs):
        for child in self.children:
            m = getattr(child, name, None)
            if m is not None:
                await m(*args, **kwargs)

    async def on_agent_start(self, context, agent):
        await self._fan("on_agent_start", context, agent)

    async def on_agent_end(self, context, agent, output):
        await self._fan("on_agent_end", context, agent, output)

    async def on_handoff(self, context, from_agent, to_agent):
        await self._fan("on_handoff", context, from_agent, to_agent)

    async def on_tool_start(self, context, agent, tool):
        await self._fan("on_tool_start", context, agent, tool)

    async def on_tool_end(self, context, agent, tool, result):
        await self._fan("on_tool_end", context, agent, tool, result)

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        await self._fan("on_llm_start", context, agent, system_prompt, input_items)

    async def on_llm_end(self, context, agent, response):
        await self._fan("on_llm_end", context, agent, response)
