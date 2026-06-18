"""Herramientas compartidas para los agentes de myexamples_utils."""

import os
from functools import lru_cache
from typing import Literal

from langchain.tools import tool, ToolRuntime

from tavily import TavilyClient

from myexamples_dto.context import Context


@lru_cache(maxsize=1)
def _get_client() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> dict:
    """Realiza una búsqueda web y devuelve los resultados."""
    return _get_client().search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


@tool
def fetch_user_data(query: str, runtime: ToolRuntime[Context]) -> str:
    """Recuperar datos para el usuario actual."""
    user_id = runtime.context.user_id
    return f"Datos para el usuario {user_id}: {query}"