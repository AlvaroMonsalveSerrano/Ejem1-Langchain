"""
Ejemplo de uso de context_schema con runtime context.

El agente recibe un contexto tipado (Context) en cada invocación.
La herramienta fetch_user_data accede a context.user_id en tiempo de
ejecución a través de ToolRuntime, sin que el LLM vea datos sensibles.

Patrón utilizado:
    agent = create_deep_agent(
        model=...,
        tools=[fetch_user_data],
        context_schema=Context,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "..."}]},
        context=Context(user_id="user-123", api_key="sk-..."),
    )

Formas de ejecutar:

    # Consulta por defecto con usuario y api-key por defecto (user-123 / sk-demo)
    > python myexamples_context/05_runtime_context.py

    # Consulta personalizada
    > python myexamples_context/05_runtime_context.py "Obtén mi actividad reciente"

    # Contexto personalizado: usuario y api-key explícitos
    > python myexamples_context/05_runtime_context.py \
          --user-id "user-456" \
          --api-key "sk-real-key" \
          "¿Cuáles son mis últimas transacciones?"

    # Solo cambio de usuario, consulta por defecto
    > python myexamples_context/05_runtime_context.py --user-id "user-789"

"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_dto.context import Context              # noqa: E402
from myexamples_utils.agent_factory import build_agent  # noqa: E402
from myexamples_utils.tools import fetch_user_data      # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agente con runtime context tipado")
    parser.add_argument(
        "query",
        nargs="?",
        default="Obtén mi actividad reciente",
        help="Consulta a enviar al agente (por defecto: 'Obtén mi actividad reciente')",
    )
    parser.add_argument(
        "--user-id",
        default="user-123",
        help="Identificador del usuario inyectado en el contexto (por defecto: user-123)",
    )
    parser.add_argument(
        "--api-key",
        default="sk-demo",
        help="API key inyectada en el contexto (por defecto: sk-demo)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    agent = build_agent(
        tools=[fetch_user_data],
        context_schema=Context,
    )

    context = Context(user_id=args.user_id, api_key=args.api_key)

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": args.query}]},
            context=context,
        )
    except Exception as exc:
        sys.exit(f"Error al invocar el agente: {exc}")

    print(f"Resultado: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
