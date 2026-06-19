"""
Ejemplo de uso de state_schema con un estado personalizado.

El agente extiende DeepAgentState con campos adicionales (page_url, file_urls)
mediante ResearchState. La herramienta cite_page accede al estado en tiempo de
ejecución a través de ToolRuntime sin necesidad de parámetros explícitos.

Patrón utilizado:
    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[cite_page],
        state_schema=ResearchState,
    )
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Cita la página actual y resume su propósito"}],
            "page_url": "https://docs.python.org/3/library/dataclasses.html",
            "file_urls": [],
        }
    )

Formas de ejecutar:

    # Consulta por defecto
    > python myexamples_context/06_custom_state_schema_context.py

    # Consulta personalizada
    > python myexamples_context/06_custom_state_schema_context.py "¿Qué página estoy visitando?"

    # Con page_url explícita
    > python myexamples_context/06_custom_state_schema_context.py \
          --page-url "https://docs.python.org" \
          "Cita la página actual"

"""

import argparse
import sys
from pathlib import Path

from deepagents import DeepAgentState, create_deep_agent
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_utils.agent_factory import build_agent  # noqa: E402
from myexamples_utils.tools import cite_page            # noqa: E402


class ResearchState(DeepAgentState):
    """Estado personalizado que extiende DeepAgentState con campos de investigación.

    Permite que las herramientas accedan a metadatos de la sesión (URL activa,
    ficheros adjuntos) a través de ToolRuntime.state, sin necesidad de que el
    LLM los reciba como parámetros de entrada.

    Attributes:
        page_url: URL de la página que el agente está procesando actualmente.
                  La herramienta cite_page la lee via runtime.state["page_url"].
        file_urls: Lista de URLs de ficheros asociados a la sesión de investigación.
                   Permite a futuras herramientas operar sobre documentos sin
                   que el usuario los repita en cada mensaje.
    """

    page_url: str
    file_urls: list[str]


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos del ejemplo.

    Returns:
        Namespace con los campos:
            query    -- consulta enviada al agente como mensaje de usuario.
            page_url -- URL inyectada en ResearchState.page_url antes de invocar
                        el agente; accesible por cite_page via ToolRuntime.state.
    """
    parser = argparse.ArgumentParser(description="Agente con estado personalizado ResearchState")
    parser.add_argument(
        "query",
        nargs="?",
        default="Cita la página actual y resume su propósito",
        help="Consulta a enviar al agente",
    )
    parser.add_argument(
        "--page-url",
        default="https://docs.python.org/3/library/dataclasses.html",
        help="URL de la página inyectada en el estado",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del ejemplo de state_schema.

    Flujo:
        1. Parsea los argumentos CLI (query, page_url).
        2. Construye el agente con cite_page y state_schema=ResearchState.
        3. Invoca el agente inyectando page_url y file_urls en el estado inicial.
        4. Imprime el contenido del último mensaje de la respuesta.

    Termina con sys.exit si el agente lanza una excepción en tiempo de ejecución.
    """
    args = parse_args()

    agent = build_agent(
        tools=[cite_page],
        state_schema=ResearchState,
    )

    try:
        result = agent.invoke(
            {
                "messages": [{"role": "user", "content": args.query}],
                "page_url": args.page_url,
                "file_urls": [],
            }
        )
    except Exception as exc:
        sys.exit(f"Error al invocar el agente: {exc}")

    print(f"Resultado: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
