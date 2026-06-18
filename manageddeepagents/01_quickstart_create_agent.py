
""" 
**Managed Deep Agents es una funcionalidad avanzada orientada a producción**, para equipos que quieren 
desplegar agentes de larga duración sin gestionar la infraestructura de runtime. Para aprender LangChain 
y LangSmith desde cero, con el plan gratuito tienes todo lo necesario: trazas, debugging, evaluación y 
el Prompt Playground. Los Managed Deep Agents son un paso mucho más adelante en la curva de aprendizaje. 
"""

import os

from managed_deepagents import Client

from dotenv import load_dotenv

# Carga las variables de entorno definidas en .env
load_dotenv()


def main() -> None:
    try:
        with Client() as client:
            agent = client.agents.create(
                name="research-assistant",
                description="Asistente de investigación con capacidad para buscar en la web y resumir fuentes.",
                model="anthropic:claude-sonnet-4-6",
                backend={"type": "state"},
                instructions=(
                    "Eres un asistente de investigación meticuloso. Busca fuentes, "
                    "toma notas y responde concisamente con las referencias correspondientes."
                ),
            )

        agent_id = agent["id"]
        print(f"Agent ID: {agent_id}")
    except Exception as ex:
        print(f"Excepción: {ex}")


if __name__ == "__main__":
    main()