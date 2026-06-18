"""
Ejemplo de uso de skills con SKILL.md.

Carga las instrucciones del agente desde skills/investigador/SKILL.md mediante
el parámetro `skills` de create_deep_agent, siguiendo el patrón de progressive
disclosure: el agente recibe nombre y descripción de la skill; lee el contenido
completo solo cuando la tarea lo requiere.

Estructura de skills esperada:
    myexamples_context/skills/
    └── investigador/
        └── SKILL.md

Formas de ejecutar:

> python myexamples_context/03_skill_context.py "Saludos desde un agente"

> python myexamples_context/03_skill_context.py

"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Añade la raíz del proyecto al path para que los imports de los paquetes
# myexamples_context y myexamples_utils funcionen con `python script.py`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_utils.agent_factory import build_agent  # noqa: E402
from myexamples_utils.tools import internet_search      # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agente investigador con skills SKILL.md")
    parser.add_argument(
        "query",
        nargs="?",
        default="Saludos desde un agente",
        help="Consulta a enviar al agente (por defecto: 'Saludos desde un agente')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skills_dir = Path(__file__).parent / "skills"

    if not skills_dir.is_dir():
        sys.exit(f"Error: directorio de skills no encontrado: {skills_dir}")

    agent = build_agent(
        tools=[internet_search],
        skills=[str(skills_dir)],
    )

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": args.query}]})
    except Exception as exc:
        sys.exit(f"Error al invocar el agente: {exc}")

    print(f"Resultado: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
