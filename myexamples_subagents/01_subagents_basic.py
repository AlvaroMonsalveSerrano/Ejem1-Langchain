"""
Ejemplo básico de subagentes con deepagents y Tavily.

Un subagente es un agente especializado que el agente principal puede delegar
tareas. La arquitectura es:

    Agente principal
    └── Subagente research-agent
            └── Herramienta internet_search (Tavily)

El agente principal recibe la query del usuario. Cuando necesita información
de internet, delega en el subagente «research-agent», que ejecuta búsquedas
reales con Tavily y devuelve los resultados al principal para sintetizarlos.

Conceptos clave:

    subagent dict -- diccionario con los campos:
        name         -- identificador único del subagente.
        description  -- cuándo debe usarlo el agente principal (orientación al LLM).
        system_prompt -- prompt de sistema del subagente; define su rol.
        tools        -- lista de funciones Python que el subagente puede llamar.
        model        -- (opcional) modelo distinto al del agente principal.

    create_deep_agent -- crea el agente principal con la lista de subagentes.

Modelos usados:
    Principal  → anthropic:claude-sonnet-4-6  (síntesis y coordinación)
    Subagente  → anthropic:claude-haiku-4-5-20251001  (búsqueda rápida y económica)

Variables de entorno necesarias (.env):
    ANTHROPIC_API_KEY  -- clave de la API de Anthropic.
    TAVILY_API_KEY     -- clave de la API de Tavily para búsquedas web.

Trazas de log:
    El módulo usa el logger «subagents» con dos niveles útiles:
    - INFO  (por defecto): query enviada, subagente configurado, duración.
    - DEBUG: trazas internas de deepagents (pasos del grafo, callbacks).

    Para activar trazas remotas en LangSmith, define en .env:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<tu clave>
        LANGCHAIN_PROJECT=<nombre del proyecto>   # opcional

Formas de ejecutar:

    # 1. Comportamiento por defecto: pregunta general con búsqueda web
    > python myexamples_subagents/01_subagents_basic.py

    # 2. Pregunta personalizada (el agente delega en el subagente si necesita buscar)
    > python myexamples_subagents/01_subagents_basic.py \
          "¿Qué es Python 3.13?"

    # 3. Pregunta de noticias (topic=news) con más resultados
    > python myexamples_subagents/01_subagents_basic.py \
          --topic news --max-results 3 \
          "Noticias IA hoy"

    # 4. Incluir contenido completo de las páginas encontradas
    > python myexamples_subagents/01_subagents_basic.py \
          --include-raw-content \
          "¿Qué es LangChain?"

    # 5. Activar trazas DEBUG locales
    > python myexamples_subagents/01_subagents_basic.py \
          --log-level DEBUG \
          "¿Qué es deepagents?"

"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Literal

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from tavily import TavilyClient

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_utils.logger import configure_logging  # noqa: E402

logger = logging.getLogger("subagents")


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos del ejemplo.

    Returns:
        Namespace con los campos:
            query            -- consulta enviada al agente principal.
            topic            -- categoría de búsqueda Tavily (general/news/finance).
            max_results      -- número máximo de resultados por búsqueda.
            include_raw_content -- si incluir el HTML completo de cada resultado.
            log_level        -- nivel de trazas de log (DEBUG, INFO, WARNING).
    """
    parser = argparse.ArgumentParser(
        description="Ejemplo básico de subagentes: agente principal + subagente de investigación con Tavily"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="¿Qué es la IA generativa?",
        help="Consulta a enviar al agente principal",
    )
    parser.add_argument(
        "--topic",
        default="general",
        choices=["general", "news", "finance"],
        help="Categoría de búsqueda Tavily (por defecto: general)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Número máximo de resultados por búsqueda Tavily (por defecto: 5)",
    )
    parser.add_argument(
        "--include-raw-content",
        action="store_true",
        help="Incluir el contenido HTML completo de cada resultado de búsqueda",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Nivel de trazas de log (por defecto: INFO)",
    )
    return parser.parse_args()


def build_search_tool(
    max_results: int,
    topic: Literal["general", "news", "finance"],
    include_raw_content: bool,
):
    """Construye la herramienta de búsqueda Tavily con los parámetros dados.

    Devuelve una función Python con la firma esperada por deepagents como tool.
    Los parámetros max_results, topic e include_raw_content quedan fijados en
    el cierre para que el subagente no necesite pasarlos en cada llamada.

    Args:
        max_results:         número máximo de resultados por búsqueda.
        topic:               categoría de búsqueda Tavily.
        include_raw_content: si incluir el HTML completo de cada resultado.

    Returns:
        Función `internet_search(query: str) -> dict` lista para usar como tool.
    """
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    def internet_search(query: str) -> dict:
        """Realiza una búsqueda web usando Tavily y devuelve los resultados."""
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

    return internet_search


def build_agent(
    max_results: int,
    topic: Literal["general", "news", "finance"],
    include_raw_content: bool,
):
    """Construye el agente principal con el subagente de investigación.

    El subagente «research-agent» recibe la herramienta de búsqueda Tavily y
    usa un modelo ligero (Haiku) para mantener bajo coste. El agente principal
    (Sonnet) coordina, sintetiza y responde al usuario.

    Args:
        max_results:         configuración que se pasa a la herramienta de búsqueda.
        topic:               categoría de búsqueda Tavily.
        include_raw_content: si incluir el HTML completo de los resultados.

    Returns:
        Agente deepagents listo para invocar.
    """
    internet_search = build_search_tool(max_results, topic, include_raw_content)

    main_model = init_chat_model("anthropic:claude-sonnet-4-6", max_tokens=4096)
    subagent_model = init_chat_model("anthropic:claude-sonnet-4-6", max_tokens=1024)

    research_subagent = {
        "name": "research-agent",
        "description": "Subagente especializado en investigación web. Úsalo cuando necesites información actualizada de internet.",
        "system_prompt": "Eres un investigador experto. Realiza búsquedas precisas, extrae la información más relevante y devuelve un resumen estructurado con las fuentes.",
        "tools": [internet_search],
        "model": subagent_model,
    }

    return create_deep_agent(
        model=main_model,
        subagents=[research_subagent],
    )


def log_token_usage(result: dict) -> None:
    """Registra el consumo de tokens acumulado en los AIMessage del resultado."""
    total_input = total_output = 0
    for msg in result.get("messages", []):
        meta = getattr(msg, "usage_metadata", None)
        if meta:
            total_input += meta.get("input_tokens", 0)
            total_output += meta.get("output_tokens", 0)
    if total_input or total_output:
        logger.info(
            "Tokens → total=%d  (input=%d, output=%d)",
            total_input + total_output,
            total_input,
            total_output,
        )


def main() -> None:
    """Punto de entrada del ejemplo de subagentes.

    Flujo:
        1. Parsea los argumentos CLI.
        2. Configura el sistema de logging.
        3. Construye la herramienta de búsqueda y el subagente con sus parámetros.
        4. Crea el agente principal referenciando el subagente.
        5. Invoca el agente con la query registrando duración y resultado.

    Termina con sys.exit si el agente lanza una excepción en tiempo de ejecución.
    """
    args = parse_args()
    configure_logging(logger, args.log_level)

    logger.info(
        "Configuración → topic=%s, max_results=%d, include_raw_content=%s",
        args.topic,
        args.max_results,
        args.include_raw_content,
    )
    logger.info("Construyendo agente principal con subagente research-agent")

    agent = build_agent(args.max_results, args.topic, args.include_raw_content)

    logger.info("Query: %s", args.query)

    t0 = time.perf_counter()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": args.query}]}
        )
    except Exception as exc:
        logger.error("Error al invocar el agente: %s", exc)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    logger.info("Invocación completada en %.2f s", elapsed)
    log_token_usage(result)

    content = result["messages"][-1].content
    logger.debug("Respuesta completa: %s", content)
    print(f"[subagents·research] Resultado:\n{content}")


if __name__ == "__main__":
    main()
