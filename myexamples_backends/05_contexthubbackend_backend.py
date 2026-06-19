"""
Ejemplo de uso de ContextHubBackend como backend del agente.

ContextHubBackend conecta el agente con el servicio Context Hub de LangSmith,
que actúa como fuente centralizada de contexto gestionado externamente: prompts
de sistema, documentos de referencia, instrucciones de comportamiento y cualquier
otro fragmento de texto que el agente debe consultar al inicio de cada sesión.

El argumento posicional de ContextHubBackend es el nombre del agente registrado
en LangSmith («my-agent» en el ejemplo). La plataforma resuelve ese nombre y
envía al agente el contexto configurado para él sin que el desarrollador tenga
que hardcodearlo en el código.

Diferencia clave respecto a otros backends:
- StateBackend   → gestiona el estado conversacional (mensajes, campos extra).
- StoreBackend   → gestiona memoria a largo plazo (clave-valor por namespace).
- FilesystemBackend → da acceso al sistema de ficheros local.
- ContextHubBackend → inyecta contexto externo desde LangSmith Context Hub.

Casos de uso habituales:
- Centralizar instrucciones de sistema en LangSmith y actualizarlas sin redesplegar.
- Proporcionar al agente documentos de referencia (políticas, FAQs) gestionados
  por equipos no técnicos desde la UI de LangSmith.
- Separar la configuración de comportamiento del agente del código fuente.
- Compartir el mismo contexto entre múltiples agentes distintos.

Requisito:
    ContextHubBackend requiere conexión a LangSmith. Define en .env:
        LANGCHAIN_API_KEY=<tu clave>
        LANGCHAIN_TRACING_V2=true          # recomendado para ver las trazas
        LANGCHAIN_PROJECT=<nombre-proyecto> # opcional

Patrón utilizado:

    from deepagents import create_deep_agent
    from deepagents.backends import ContextHubBackend

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        backend=ContextHubBackend("my-agent"),
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "¿Qué instrucciones tienes?"}]}
    )

Parámetros de ContextHubBackend:
    agent_name -- nombre del agente registrado en LangSmith Context Hub.
                  La plataforma resuelve este nombre y entrega el contexto
                  configurado para él (prompts, documentos, instrucciones).
                  Debe existir previamente en LangSmith; de lo contrario la
                  invocación falla con un error de autenticación o de recurso
                  no encontrado.

Trazas de log:
    El módulo usa el logger «contexthubbackend» con dos niveles útiles:
    - INFO  (por defecto): agente construido, nombre del agente en Context Hub,
            query enviada, duración de la invocación y resultado.
    - DEBUG: trazas internas de LangChain/deepagents (pasos del grafo, callbacks,
            peticiones HTTP al Context Hub de LangSmith).

    Para activar trazas remotas en LangSmith, define en .env:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<tu clave>
        LANGCHAIN_PROJECT=<nombre del proyecto>   # opcional

Formas de ejecutar:

    # 1. Consulta por defecto: el agente responde con el contexto de «my-agent»
    > python myexamples_backends/05_contexthubbackend_backend.py

    # 2. Preguntar al agente qué instrucciones tiene configuradas en Context Hub
    > python myexamples_backends/05_contexthubbackend_backend.py \
          "¿Qué instrucciones o contexto tienes configurado?"

    # 3. Usar un agente distinto registrado en LangSmith
    > python myexamples_backends/05_contexthubbackend_backend.py \
          --agent-name "soporte-tecnico" \
          "¿Cuál es tu propósito?"

    # 4. Verificar que el contexto externo se aplica correctamente
    > python myexamples_backends/05_contexthubbackend_backend.py \
          --agent-name "my-agent" \
          "Resume en una frase el contexto que has recibido"

    # 5. Activar trazas DEBUG para ver las peticiones al Context Hub
    > python myexamples_backends/05_contexthubbackend_backend.py \
          --log-level DEBUG \
          "¿Qué instrucciones tienes?"

"""

import argparse
import logging
import sys
import time
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import ContextHubBackend
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_utils.logger import configure_logging  # noqa: E402

logger = logging.getLogger("contexthubbackend")

_DEFAULT_AGENT_NAME = "my-agent"


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos del ejemplo.

    Returns:
        Namespace con los campos:
            query      -- consulta enviada al agente como mensaje de usuario.
            agent_name -- nombre del agente registrado en LangSmith Context Hub.
                          Determina qué contexto externo (prompts, documentos,
                          instrucciones) recibe el agente al iniciar la sesión.
            log_level  -- nivel de trazas de log («DEBUG», «INFO», «WARNING»).
    """
    parser = argparse.ArgumentParser(
        description="Ejemplo de ContextHubBackend con contexto gestionado en LangSmith"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="¿Qué instrucciones tienes configuradas?",
        help="Consulta a enviar al agente",
    )
    parser.add_argument(
        "--agent-name",
        default=_DEFAULT_AGENT_NAME,
        help=f"Nombre del agente en LangSmith Context Hub (por defecto: «{_DEFAULT_AGENT_NAME}»)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Nivel de trazas de log (por defecto: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del ejemplo de ContextHubBackend.

    Flujo:
        1. Parsea los argumentos CLI (query, --agent-name, --log-level).
        2. Configura el sistema de logging con el nivel indicado.
        3. Construye el agente con ContextHubBackend apuntando al agente
           registrado en LangSmith Context Hub.
        4. Invoca el agente registrando inicio, duración y resultado.
        5. Imprime el contenido del último mensaje de la respuesta.

    ContextHubBackend requiere LANGCHAIN_API_KEY definida en .env para autenticar
    las peticiones al servicio Context Hub de LangSmith. Si la clave no está
    presente o el agente no existe en Context Hub, la invocación falla con un
    error de autenticación o de recurso no encontrado.

    Termina con sys.exit si el agente lanza una excepción en tiempo de ejecución.
    """
    args = parse_args()
    configure_logging(logger, args.log_level)

    logger.info(
        "Construyendo agente con ContextHubBackend (agent_name=%s)", args.agent_name
    )

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        backend=ContextHubBackend(args.agent_name),
    )

    logger.info("Context Hub agent: %s", args.agent_name)
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

    content = result["messages"][-1].content
    logger.debug("Respuesta completa: %s", content)
    print(f"[ContextHubBackend · agent={args.agent_name}] Resultado: {content}")


if __name__ == "__main__":
    main()
