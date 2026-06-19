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
    state_schema: type | None = None,
) -> CompiledStateGraph:
    """Crea y devuelve un agente deepagents con la configuración indicada.

    Envuelve create_deep_agent centralizando los parámetros opcionales: solo
    los que reciben un valor distinto de None/vacío se reenvían, evitando
    colisiones con los defaults internos de deepagents.

    Args:
        model:          Identificador del modelo en formato «proveedor:nombre»
                        (p. ej. «anthropic:claude-sonnet-4-6»).
        tools:          Lista de funciones decoradas con @tool que el agente
                        puede invocar durante la conversación.
        memory:         Lista de claves de memoria persistente entre sesiones.
        skills:         Lista de rutas a directorios que contienen ficheros
                        SKILL.md con instrucciones de habilidades del agente.
        context_schema: Tipo (dataclass) que define el contexto tipado inyectado
                        en cada invocación. Las herramientas lo leen via
                        ToolRuntime[T].context sin que el LLM lo vea.
        state_schema:   Subclase de DeepAgentState que amplía el estado del
                        grafo con campos adicionales accesibles por las
                        herramientas via ToolRuntime.state.

    Returns:
        Grafo compilado de LangGraph listo para invocar con agent.invoke().
    """
    kwargs: dict[str, Any] = dict(model=model, tools=tools or [], memory=memory or [])
    if skills:
        kwargs["skills"] = skills
    if context_schema is not None:
        kwargs["context_schema"] = context_schema
    if state_schema is not None:
        kwargs["state_schema"] = state_schema
    return create_deep_agent(**kwargs)
