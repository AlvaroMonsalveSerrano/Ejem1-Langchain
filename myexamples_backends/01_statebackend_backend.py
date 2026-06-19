"""
Ejemplo de uso de StateBackend como backend explícito del agente.

StateBackend es el backend por defecto que deepagents usa internamente cuando
no se especifica ningún backend. Este ejemplo muestra ambas formas equivalentes
de crear un agente: sin backend (usa StateBackend implícitamente) y con
StateBackend declarado de forma explícita.

Trazas de log:
    El módulo usa el logger «statebackend» con dos niveles útiles:
    - INFO  (por defecto): muestra qué agente se selecciona, cuándo empieza y
            termina la invocación, y el resultado final.
    - DEBUG: añade trazas internas de LangChain/deepagents (tokens, callbacks,
            pasos del grafo) activando el logger raíz de langchain.

    Para activar trazas remotas en LangSmith, define en .env:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<tu clave>
        LANGCHAIN_PROJECT=<nombre del proyecto>   # opcional

Patrón utilizado:

    # Por defecto, deepagents usa StateBackend internamente
    agent = create_deep_agent(model="anthropic:claude-sonnet-4-6")

    # Equivalente explícito: declarar StateBackend como backend
    agent2 = create_deep_agent(
        model="anthropic:claude-haiku-4-5-20251001",
        backend=StateBackend(),
    )

Formas de ejecutar:

    # Consulta por defecto con el agente implícito (StateBackend automático)
    > python myexamples_backends/01_statebackend_backend.py

    # Consulta personalizada
    > python myexamples_backends/01_statebackend_backend.py "¿Qué es LangChain?"

    # Usar el agente con StateBackend explícito
    > python myexamples_backends/01_statebackend_backend.py --explicit "¿Cuál es la capital de Francia?"

    # Activar trazas DEBUG locales (pasos internos del grafo y callbacks)
    > python myexamples_backends/01_statebackend_backend.py --log-level DEBUG "¿Qué es LangChain?"

"""

import argparse
import logging
import sys
import time
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_utils.logger import configure_logging  # noqa: E402

logger = logging.getLogger("statebackend")


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos del ejemplo.

    Returns:
        Namespace con los campos:
            query     -- consulta enviada al agente.
            explicit  -- si está presente, usa el agente con StateBackend explícito
                         en lugar del agente con backend por defecto.
            log_level -- nivel de log local («DEBUG», «INFO», «WARNING»).
    """
    parser = argparse.ArgumentParser(description="Ejemplo de StateBackend implícito y explícito")
    parser.add_argument(
        "query",
        nargs="?",
        default="Explica en una frase qué es un StateBackend en deepagents",
        help="Consulta a enviar al agente",
    )
    parser.add_argument(
        "--explicit",
        action="store_true",
        help="Usa el agente con StateBackend declarado explícitamente",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Nivel de trazas de log (por defecto: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del ejemplo de StateBackend.

    Flujo:
        1. Parsea los argumentos CLI (query, --explicit, --log-level).
        2. Configura el sistema de logging con el nivel indicado.
        3. Construye ambos agentes (implícito y explícito) y selecciona el activo.
        4. Invoca el agente registrando inicio, duración y resultado.
        5. Imprime el contenido del último mensaje de la respuesta.

    Ambos agentes son funcionalmente equivalentes: deepagents usa StateBackend
    internamente cuando no se especifica backend. El flag --explicit ilustra
    que declararlo no cambia el comportamiento.

    Termina con sys.exit si el agente lanza una excepción en tiempo de ejecución.
    """
    args = parse_args()
    configure_logging(logger, args.log_level)

    # Por defecto, deepagents proporciona StateBackend automáticamente
    logger.debug("Construyendo agente implícito (anthropic:claude-sonnet-4-6)")
    agent = create_deep_agent(model="anthropic:claude-sonnet-4-6")

    # Equivalente explícito: mismo comportamiento, backend declarado
    logger.debug("Construyendo agente explícito (anthropic:claude-haiku-4-5-20251001, StateBackend)")
    agent2 = create_deep_agent(
        model="anthropic:claude-haiku-4-5-20251001",
        backend=StateBackend(),
    )

    selected = agent2 if args.explicit else agent
    label = "StateBackend explícito" if args.explicit else "StateBackend implícito (por defecto)"

    logger.info("Agente seleccionado: %s", label)
    logger.info("Query: %s", args.query)

    t0 = time.perf_counter()
    try:
        result = selected.invoke(
            {"messages": [{"role": "user", "content": args.query}]}
        )
    except Exception as exc:
        logger.error("Error al invocar el agente (%s): %s", label, exc)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    logger.info("Invocación completada en %.2f s", elapsed)

    content = result["messages"][-1].content
    logger.debug("Respuesta completa: %s", content)
    print(f"[{label}] Resultado: {content}")


if __name__ == "__main__":
    main()
