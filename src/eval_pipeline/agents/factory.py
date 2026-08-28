from __future__ import annotations

from typing import Any

from eval_pipeline.agents.prompts.oh_system import load_oh_system_prompt
from eval_pipeline.agents.tools.oh import make_tools
from eval_pipeline.utils.docker import DockerExecutor


def build_agent(
    *,
    llm: Any,
    system_prompt: str,
    tools: list[Any],
    stop_at: list[str] | None = None,
    agent_name: str = "agent",
):
    from agents import Agent

    return Agent(
        name=agent_name,
        instructions=system_prompt,
        model=llm,
        tools=tools,
        tool_use_behavior={
            "stop_at_tool_names": stop_at or ["submit"],
        },
    )


_ALL_BASE_TOOL_NAMES = ("bash", "str_replace_editor", "submit")


def build_openhands_agent(
    *,
    docker: DockerExecutor,
    work_dir: str,
    llm_model: Any,
    agent_name: str = "CodeActAgent",
    system_prompt: str | None = None,
    extra_tools: list[Any] | None = None,
    stop_at_tool_names: list[str] | None = None,
    max_observation_length: int = 16000,
    bash_timeout: int = 1800,
    base_tools: list[str] | None = None,
) -> tuple[Any, list[Any]]:
    if base_tools is None:
        base_tool_names: tuple[str, ...] = _ALL_BASE_TOOL_NAMES
    else:
        for name in base_tools:
            if name not in _ALL_BASE_TOOL_NAMES:
                raise ValueError(
                    f"build_openhands_agent: unknown base tool {name!r}; "
                    f"valid names are {_ALL_BASE_TOOL_NAMES}"
                )
        base_tool_names = tuple(base_tools)

    all_base = make_tools(
        docker, work_dir,
        max_observation_length=max_observation_length,
        bash_timeout=bash_timeout,
    )
    name_to_tool = {getattr(t, "name", None): t for t in all_base}
    selected = [name_to_tool[n] for n in base_tool_names]
    tools = selected + list(extra_tools or [])

    prompt = system_prompt or load_oh_system_prompt()
    agent = build_agent(
        llm=llm_model,
        system_prompt=prompt,
        tools=tools,
        stop_at=stop_at_tool_names,
        agent_name=agent_name,
    )
    return agent, tools
