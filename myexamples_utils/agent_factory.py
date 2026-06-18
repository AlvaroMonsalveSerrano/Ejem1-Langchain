"""Factoría para crear instancias de agentes deepagents."""

from collections.abc import Callable
from typing import Any

from deepagents import create_deep_agent
from langgraph.graph.state import CompiledStateGraph


def build_agent(
    model: str = "anthropic:claude-sonnet-4-6",
    tools: list[Callable] | None = None,
    memory: list[str] | None = None,
    skills: list[str] | None = None,
    context_schema: type | None = None,
) -> CompiledStateGraph:
    """Crea y devuelve un agente deepagents con la configuración indicada."""
    kwargs: dict[str, Any] = dict(model=model, tools=tools or [], memory=memory or [])
    if skills:
        kwargs["skills"] = skills
    if context_schema is not None:
        kwargs["context_schema"] = context_schema
    return create_deep_agent(**kwargs)
