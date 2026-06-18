"""
Ejemplo 01 — Visión general de deepagents
==========================================
Demuestra la creación y uso básico de un agente con deepagents:
  - Carga de credenciales desde .env (ANTHROPIC_API_KEY).
  - Definición de una herramienta personalizada (get_weather).
  - Creación del agente con create_deep_agent apuntando a Claude Sonnet.
  - Invocación del agente y extracción legible de la respuesta final.
"""

from dotenv import load_dotenv
from deepagents import create_deep_agent

# Carga las variables de entorno definidas en .env
load_dotenv()


def get_weather(city: str) -> str:
    """Devuelve el tiempo actual para una ciudad dada."""
    return f"¡En {city} siempre hace sol!"


def main() -> None:
    """Crea el agente, lanza una consulta y muestra la respuesta por consola."""
    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[get_weather],
        system_prompt="Eres un asistente útil",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "¿qué tiempo hace en Madrid?"}]}
    )

    # El último mensaje de la lista es siempre la respuesta final del agente.
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
