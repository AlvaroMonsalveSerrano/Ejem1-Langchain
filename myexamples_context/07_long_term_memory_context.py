"""
Ejemplo de uso de memoria a largo plazo con store e InMemoryStore.

El agente persiste preferencias del usuario entre conversaciones mediante
StoreBackend montado en «/memories/». El system_prompt le indica cuándo y
dónde guardar los recuerdos; make_backend conecta StateBackend (estado
conversacional) y StoreBackend (memoria persistente) bajo un CompositeBackend.

Patrón utilizado:
    agent = create_deep_agent(
        model="google_genai:gemini-3.5-flash",
        store=InMemoryStore(),
        backend=make_backend,
        system_prompt=\"\"\"Cuando los usuarios te indiquen sus preferencias, guárdalas en
        /memories/user_preferences.txt para recordarlas en conversaciones futuras.\"\"\",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Me gusta el café solo sin azúcar"}]}
    )

Formas de ejecutar:

    # 1. Preferencia por defecto: el agente guarda que al usuario le gusta el café solo
    > python myexamples_context/07_long_term_memory_context.py

    # 2. Guardar una preferencia de formato de respuesta
    #    El agente persiste esta instrucción en /memories/user_preferences.txt
    > python myexamples_context/07_long_term_memory_context.py "Prefiero respuestas cortas y directas"

    # 3. Guardar una preferencia de idioma y estilo
    > python myexamples_context/07_long_term_memory_context.py \
          "Respóndeme siempre en español y con ejemplos de código"

    # 4. Guardar varias preferencias en un mismo mensaje
    > python myexamples_context/07_long_term_memory_context.py \
          "Soy desarrollador Python, prefiero ejemplos con type hints y sin comentarios obvios"

    # 5. Consultar si el agente recuerda las preferencias guardadas
    #    (útil para verificar que StoreBackend ha persistido el fichero)
    > python myexamples_context/07_long_term_memory_context.py \
          "¿Qué preferencias mías tienes guardadas?"

    NOTA: InMemoryStore persiste solo durante el proceso. Para memoria entre
    ejecuciones distintas se necesita un backend persistente (p. ej. Redis o SQLite).

"""

import sys
import argparse
from pathlib import Path

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langgraph.store.memory import InMemoryStore

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_utils.backend import make_backend  # noqa: E402

SYSTEM_PROMPT = """\
Cuando los usuarios te indiquen sus preferencias, guárdalas en \
/memories/user_preferences.txt para recordarlas en conversaciones futuras.\
"""


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos del ejemplo.

    Returns:
        Namespace con el campo:
            query -- mensaje del usuario enviado al agente. Si contiene una
                     preferencia, el agente la persiste en /memories/user_preferences.txt
                     para recuperarla en conversaciones futuras.
    """
    parser = argparse.ArgumentParser(description="Agente con memoria a largo plazo")
    parser.add_argument(
        "query",
        nargs="?",
        default="Me gusta el café solo sin azúcar",
        help="Mensaje del usuario (por defecto: 'Me gusta el café solo sin azúcar')",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del ejemplo de memoria a largo plazo.

    Flujo:
        1. Parsea los argumentos CLI (query).
        2. Construye el agente con InMemoryStore, make_backend y system_prompt.
        3. Invoca el agente con el mensaje del usuario.
        4. Imprime el contenido del último mensaje de la respuesta.

    El system_prompt instruye al agente a guardar las preferencias del usuario
    en /memories/user_preferences.txt a través de StoreBackend.
    Termina con sys.exit si el agente lanza una excepción en tiempo de ejecución.
    """
    args = parse_args()

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        store=InMemoryStore(),
        backend=make_backend,
        system_prompt=SYSTEM_PROMPT,
    )

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": args.query}]}
        )
    except Exception as exc:
        sys.exit(f"Error al invocar el agente: {exc}")

    print(f"Resultado: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
