"""
Ejemplo de uso de memory con AGENTS.md.

Carga las instrucciones del agente desde AGENTS.md mediante el parámetro
`memory` de create_deep_agent, en lugar de un system_prompt hardcodeado.

Formas de ejecutar:

> python myexamples_context/02_memory_context.py "Saludos desde un agente"

>python myexamples_context/02_memory_context.py

"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_utils.agent_factory import build_agent  # noqa: E402
from myexamples_utils.tools import internet_search      # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agente investigador con memoria AGENTS.md")
    parser.add_argument(
        "query",
        nargs="?",
        default="Saludos desde un agente",
        help="Consulta a enviar al agente (por defecto: 'Saludos desde un agente')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agents_md = str(Path(__file__).parent / "AGENTS.md")

    agent = build_agent(
        tools=[internet_search],
        memory=[agents_md],
    )

    result = agent.invoke({"messages": [{"role": "user", "content": args.query}]})
    print(f"Resultado: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
