import os
from typing import Literal
from dotenv import load_dotenv

from tavily import TavilyClient
from deepagents import create_deep_agent

# Carga las variables de entorno definidas en .env
load_dotenv()

#
# Definición de la herramienta de búsqueda en Internet
#

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Realiza una búsqueda web"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )



def main() -> None:

    # El sistema solicita al agente que se convierta en un investigador experto.
    research_instructions = """Eres un investigador experto. Tu trabajo consiste en realizar una investigación exhaustiva y luego redactar un informe impecable.

    Tienes acceso a una herramienta de búsqueda en internet como principal medio para recopilar información.

    ## `internet_search`

    Utilice esta función para realizar una búsqueda en internet con una consulta específica. Puede especificar el número máximo de resultados, el tema y si se debe incluir el contenido sin formato.
    """

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[internet_search],
        system_prompt=research_instructions,
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "Qué es langgraph?"}]})

    # Imprime la respuesta del agente.
    print(f"Resultado: {result['messages'][-1].content}")



if __name__ == "__main__":
    main()