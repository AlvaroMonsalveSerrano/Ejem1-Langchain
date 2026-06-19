"""
Ejemplo de uso de CompositeBackend como backend del agente.

CompositeBackend combina múltiples backends en uno solo mediante un sistema de
enrutamiento por prefijos de ruta. Permite que un mismo agente acceda a distintos
mecanismos de almacenamiento dependiendo del tipo de operación que necesita realizar,
sin necesidad de crear agentes distintos para cada backend.

El parámetro «default» define el backend que maneja todas las operaciones que no
coinciden con ninguna ruta explícita. El parámetro «routes» es un diccionario
que mapea prefijos de ruta (strings con formato «/prefijo/») al backend que debe
gestionar las operaciones dirigidas a esa ruta.

En el ejemplo canónico:
- StateBackend() como «default» gestiona el estado conversacional del grafo (mensajes,
  flujo del agente). Es el comportamiento por defecto en cualquier petición sin ruta.
- StoreBackend(namespace=...) bajo «/memories/» gestiona la memoria a largo plazo
  (clave-valor) cuando el agente accede a ese espacio de nombres.

El store (InMemoryStore) se pasa a create_deep_agent, NO al backend. Esto es
importante: CompositeBackend y StoreBackend declaran cómo se usa el store, pero
la instancia del store la provee create_deep_agent.

Diferencia clave respecto a backends individuales:
- StateBackend   → estado conversacional (default de CompositeBackend aquí).
- StoreBackend   → memoria clave-valor por namespace (ruta «/memories/» aquí).
- CompositeBackend → orquesta ambos: conversación + memoria en un único agente.

Casos de uso habituales:
- Agente que mantiene conversación (StateBackend) y recuerda datos entre mensajes
  (StoreBackend), todo en una sola instancia.
- Separar el espacio de estado conversacional del espacio de memoria persistente.
- Escalar fácilmente añadiendo rutas adicionales (p. ej. «/files/» → FilesystemBackend).
- Sustituir InMemoryStore por un store persistente en producción sin cambiar código.

Patrón utilizado:

    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
    from langgraph.store.memory import InMemoryStore

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        backend=CompositeBackend(
            default=StateBackend(),
            routes={
                "/memories/": StoreBackend(namespace=lambda _rt: ("memories",)),
            },
        ),
        store=InMemoryStore(),  # Store pasado a create_deep_agent, no al backend
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Recuerda que prefiero respuestas cortas"}]}
    )

Parámetros de CompositeBackend:
    default -- backend que gestiona todas las operaciones sin ruta explícita.
               Habitualmente StateBackend() para mantener el flujo conversacional.
    routes  -- diccionario {prefijo_ruta: backend} que enruta operaciones dirigidas
               a ese prefijo al backend correspondiente. El prefijo debe tener
               formato «/nombre/» (barra inicial y final).

Parámetros de StoreBackend en routes:
    namespace -- función ``(runtime) -> tuple[str, ...]`` que define la partición
                 del store. Con ``lambda _rt: ("memories",)`` todos los usuarios
                 comparten el mismo namespace; para aislar por usuario se usaría
                 ``lambda rt: (rt.server_info.user.identity, "memories")``.

Parámetros de create_deep_agent relevantes:
    store     -- instancia del store subyacente. InMemoryStore() es válido para
                 desarrollo local; en LangSmith Deployment se omite y la
                 plataforma provee el store persistente.

Trazas de log:
    El módulo usa el logger «compositebackend» con dos niveles útiles:
    - INFO  (por defecto): agente construido, backends activos, query enviada,
            duración de la invocación y resultado.
    - DEBUG: trazas internas de LangChain/deepagents (pasos del grafo, callbacks,
            operaciones de enrutamiento entre backends).

    Para activar trazas remotas en LangSmith, define en .env:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<tu clave>
        LANGCHAIN_PROJECT=<nombre del proyecto>   # opcional

Formas de ejecutar:

    # 1. Consulta por defecto: el agente responde con StateBackend (ruta por defecto)
    > python myexamples_backends/06_compositebackend_backend.py

    # 2. Pedir al agente que guarde una preferencia en el store de memorias
    > python myexamples_backends/06_compositebackend_backend.py \
          "Recuerda que prefiero respuestas en forma de lista numerada"

    # 3. Verificar que el agente mantiene el hilo conversacional (StateBackend)
    > python myexamples_backends/06_compositebackend_backend.py \
          "¿Qué backends tienes disponibles y para qué sirve cada uno?"

    # 4. Consultar la memoria guardada (acceso al StoreBackend vía ruta /memories/)
    > python myexamples_backends/06_compositebackend_backend.py \
          "¿Qué preferencias tienes guardadas sobre mí en tu memoria?"

    # 5. Activar trazas DEBUG para ver el enrutamiento entre backends
    > python myexamples_backends/06_compositebackend_backend.py \
          --log-level DEBUG \
          "Guarda en memoria que soy desarrollador Python"

    # 6. Combinar memoria y conversación en la misma sesión
    > python myexamples_backends/06_compositebackend_backend.py \
          "Primero guarda en memoria que soy experto en LangChain, luego dime qué recuerdas"

"""

import argparse
import logging
import sys
import time
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from dotenv import load_dotenv
from langgraph.store.memory import InMemoryStore

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_utils.logger import configure_logging  # noqa: E402

logger = logging.getLogger("compositebackend")


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos del ejemplo.

    Returns:
        Namespace con los campos:
            query     -- consulta enviada al agente como mensaje de usuario.
            log_level -- nivel de trazas de log («DEBUG», «INFO», «WARNING»).
    """
    parser = argparse.ArgumentParser(
        description="Ejemplo de CompositeBackend: StateBackend + StoreBackend en un único agente"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="¿Qué puedes hacer y qué backends tienes configurados?",
        help="Consulta a enviar al agente",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Nivel de trazas de log (por defecto: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del ejemplo de CompositeBackend.

    Flujo:
        1. Parsea los argumentos CLI (query, --log-level).
        2. Configura el sistema de logging con el nivel indicado.
        3. Construye el agente con CompositeBackend:
           - default=StateBackend() para el flujo conversacional.
           - routes={"/memories/": StoreBackend(...)} para memoria clave-valor.
           - store=InMemoryStore() pasado a create_deep_agent como store subyacente.
        4. Invoca el agente registrando inicio, duración y resultado.
        5. Imprime el contenido del último mensaje de la respuesta.

    El enrutamiento es interno al framework: el agente decide qué backend usar
    según el tipo de operación. Las queries del usuario siempre se envían como
    mensajes estándar; es el grafo interno quien decide si acceder al store de
    memorias (/memories/) o al estado conversacional (default).

    InMemoryStore persiste solo durante el proceso. Para memoria entre ejecuciones
    distintas se necesita un store persistente o LangSmith Deployment (omitiendo
    el parámetro store en create_deep_agent).

    Termina con sys.exit si el agente lanza una excepción en tiempo de ejecución.
    """
    args = parse_args()
    configure_logging(logger, args.log_level)

    logger.info(
        "Construyendo agente con CompositeBackend "
        "(default=StateBackend, routes={/memories/: StoreBackend})"
    )

    store = InMemoryStore()

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        backend=CompositeBackend(
            default=StateBackend(),
            routes={
                "/memories/": StoreBackend(namespace=lambda _rt: ("memories",)),
            },
        ),
        store=store,
    )

    logger.info("Store: InMemoryStore (local, persiste durante el proceso)")
    logger.info("Backends: default=StateBackend · /memories/=StoreBackend")
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
    print(f"[CompositeBackend · StateBackend + StoreBackend] Resultado: {content}")


if __name__ == "__main__":
    main()
