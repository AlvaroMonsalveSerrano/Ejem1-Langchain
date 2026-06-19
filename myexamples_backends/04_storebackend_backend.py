"""
Ejemplo de uso de StoreBackend como backend del agente.

StoreBackend conecta al agente con un store de clave-valor (p. ej. InMemoryStore)
y le permite leer y escribir entradas con persistencia durante la vida del proceso.
A diferencia de StateBackend (que gestiona el estado conversacional del grafo),
StoreBackend proporciona memoria a largo plazo dentro de una sesión: el agente
puede guardar preferencias, notas o resultados intermedios y recuperarlos en
mensajes posteriores de la misma ejecución.

El parámetro namespace es una función que recibe el runtime y devuelve una tupla
de cadenas que actúa como partición del store. Usando la identidad del usuario
(rt.server_info.user.identity) cada usuario obtiene un espacio de memoria aislado,
evitando que los datos de una sesión contaminen los de otra.

Para entornos de producción (LangSmith Deployment) se omite store y la plataforma
gestiona la persistencia; en desarrollo local se usa InMemoryStore().

Casos de uso habituales:
- Guardar preferencias del usuario y recuperarlas en el mismo proceso.
- Acumular resultados intermedios entre varios mensajes de la misma sesión.
- Aislar la memoria de cada usuario mediante namespace dinámico.
- Migrar de InMemoryStore (local) a store persistente (producción) sin cambiar código.

Patrón utilizado:

    from deepagents import create_deep_agent
    from deepagents.backends import StoreBackend
    from langgraph.store.memory import InMemoryStore

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        backend=StoreBackend(
            namespace=lambda rt: (rt.server_info.user.identity,),
        ),
        store=InMemoryStore(),  # En desarrollo local; omitir en LangSmith Deployment
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Recuerda que prefiero respuestas cortas"}]}
    )

Parámetros de StoreBackend:
    namespace -- función ``(runtime) -> tuple[str, ...]`` que devuelve la clave
                 de partición del store. Cada combinación de valores crea un
                 espacio de memoria independiente. Usar la identidad del usuario
                 garantiza aislamiento entre sesiones.

Parámetros de create_deep_agent relevantes:
    store     -- instancia del store subyacente. InMemoryStore() es válido para
                 desarrollo local; en LangSmith Deployment se omite y la
                 plataforma provee el store persistente.

Trazas de log:
    El módulo usa el logger «storebackend» con dos niveles útiles:
    - INFO  (por defecto): agente construido, identidad de usuario, query enviada,
            duración de la invocación y resultado.
    - DEBUG: trazas internas de LangChain/deepagents (pasos del grafo, callbacks,
            operaciones de lectura/escritura en el store).

    Para activar trazas remotas en LangSmith, define en .env:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<tu clave>
        LANGCHAIN_PROJECT=<nombre del proyecto>   # opcional

Formas de ejecutar:

    # 1. Consulta por defecto: el agente guarda una preferencia en el store
    > python myexamples_backends/04_storebackend_backend.py

    # 2. Guardar una preferencia personalizada
    > python myexamples_backends/04_storebackend_backend.py \
          "Recuerda que prefiero respuestas en forma de lista numerada"

    # 3. Verificar que el agente ha guardado la preferencia
    #    (en la misma ejecución, el store persiste en memoria)
    > python myexamples_backends/04_storebackend_backend.py \
          "¿Qué preferencias tienes guardadas sobre mí?"

    # 4. Simular un usuario diferente con namespace propio
    #    Cada identidad tiene su espacio de memoria aislado
    > python myexamples_backends/04_storebackend_backend.py \
          --user-id "usuario_b" \
          "Recuerda que soy desarrollador backend y prefiero ejemplos en Python"

    # 5. Activar trazas DEBUG para ver las operaciones en el store
    > python myexamples_backends/04_storebackend_backend.py \
          --log-level DEBUG \
          "Recuerda que prefiero respuestas cortas"

"""

import argparse
import logging
import sys
import time
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import StoreBackend
from dotenv import load_dotenv
from langgraph.store.memory import InMemoryStore

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_utils.logger import configure_logging  # noqa: E402

logger = logging.getLogger("storebackend")

_DEFAULT_USER_ID = "usuario_demo"


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos del ejemplo.

    Returns:
        Namespace con los campos:
            query     -- consulta enviada al agente como mensaje de usuario.
            user_id   -- identidad del usuario usada como namespace del store.
                         Cada user_id obtiene un espacio de memoria aislado en
                         StoreBackend, evitando colisiones entre sesiones.
            log_level -- nivel de trazas de log («DEBUG», «INFO», «WARNING»).
    """
    parser = argparse.ArgumentParser(
        description="Ejemplo de StoreBackend con namespace por usuario e InMemoryStore"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Recuerda que prefiero respuestas cortas y directas",
        help="Consulta a enviar al agente",
    )
    parser.add_argument(
        "--user-id",
        default=_DEFAULT_USER_ID,
        help=f"Identidad del usuario para el namespace del store (por defecto: «{_DEFAULT_USER_ID}»)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Nivel de trazas de log (por defecto: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del ejemplo de StoreBackend.

    Flujo:
        1. Parsea los argumentos CLI (query, --user-id, --log-level).
        2. Configura el sistema de logging con el nivel indicado.
        3. Construye el agente con StoreBackend usando el user_id como namespace
           y un InMemoryStore como store subyacente.
        4. Invoca el agente registrando inicio, duración y resultado.
        5. Imprime el contenido del último mensaje de la respuesta.

    El namespace ``lambda rt: (rt.server_info.user.identity,)`` particiona el
    store por identidad de usuario: cada user_id accede únicamente a sus propias
    entradas, sin visibilidad sobre las de otros usuarios.

    InMemoryStore persiste solo durante el proceso; para memoria entre
    ejecuciones distintas se necesita un store persistente (p. ej. Redis o SQLite)
    o un LangSmith Deployment (omitiendo el parámetro store).

    Termina con sys.exit si el agente lanza una excepción en tiempo de ejecución.
    """
    args = parse_args()
    configure_logging(logger, args.log_level)

    logger.info("Construyendo agente con StoreBackend (user_id=%s)", args.user_id)

    store = InMemoryStore()

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        backend=StoreBackend(
            namespace=lambda rt: (rt.server_info.user.identity,),
        ),
        store=store,
    )

    logger.info("Store: InMemoryStore (local, persiste durante el proceso)")
    logger.info("Namespace activo: (%s,)", args.user_id)
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
    print(f"[StoreBackend · user={args.user_id}] Resultado: {content}")


if __name__ == "__main__":
    main()
