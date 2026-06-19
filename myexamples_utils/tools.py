"""Herramientas compartidas para los agentes de myexamples_utils."""

import os
from functools import lru_cache
from typing import Literal

from langchain.tools import tool, ToolRuntime

from tavily import TavilyClient

from myexamples_dto.context import Context


@lru_cache(maxsize=1)
def _get_client() -> TavilyClient:
    """Devuelve la instancia singleton de TavilyClient.

    El decorador lru_cache(maxsize=1) garantiza que el cliente se construye
    una sola vez por proceso, reutilizando la conexión en llamadas sucesivas.
    Lee TAVILY_API_KEY del entorno; falla en KeyError si no está definida.
    """
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> dict:
    """Realiza una búsqueda web mediante Tavily y devuelve los resultados.

    Herramienta de propósito general para que los agentes consulten información
    actualizada en internet. Soporta búsqueda de noticias y datos financieros
    además de búsqueda general.

    Args:
        query:               Texto de la consulta a buscar.
        max_results:         Número máximo de resultados a devolver (por defecto 5).
        topic:               Tipo de búsqueda: «general», «news» o «finance».
        include_raw_content: Si es True, incluye el HTML/texto completo de cada
                             resultado además del fragmento resumido.

    Returns:
        Diccionario con la respuesta de Tavily, incluyendo la clave «results»
        con la lista de documentos encontrados.
    """
    return _get_client().search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


@tool
def fetch_user_data(query: str, runtime: ToolRuntime[Context]) -> str:
    """Recupera datos asociados al usuario autenticado en la sesión actual.

    Accede a Context.user_id a través de ToolRuntime en tiempo de ejecución,
    sin que el LLM reciba ni vea el identificador como parámetro de entrada.
    El agente debe haberse construido con context_schema=Context para que
    runtime.context esté disponible.

    Args:
        query:   Consulta o tipo de datos a recuperar para el usuario.
        runtime: Contexto de ejecución inyectado automáticamente por deepagents.
                 Proporciona acceso a runtime.context (instancia de Context).

    Returns:
        Cadena con los datos recuperados para el usuario identificado por
        runtime.context.user_id.
    """
    user_id = runtime.context.user_id
    return f"Datos para el usuario {user_id}: {query}"


@tool
def cite_page(runtime: ToolRuntime) -> str:
    """Devuelve la URL de la página que el agente está procesando actualmente.

    Lee page_url directamente del estado del grafo via ToolRuntime.state,
    sin exponerla al LLM como parámetro. El agente debe haberse construido
    con state_schema=ResearchState (o equivalente) para que el campo
    page_url esté disponible en el estado.

    Args:
        runtime: Contexto de ejecución inyectado automáticamente por deepagents.
                 Proporciona acceso a runtime.state, el estado completo del grafo.

    Returns:
        URL de la página actual almacenada en runtime.state["page_url"].
    """
    return runtime.state["page_url"]
