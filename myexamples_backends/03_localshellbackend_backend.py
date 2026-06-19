"""
Ejemplo de uso de LocalShellBackend como backend del agente.

LocalShellBackend permite al agente ejecutar comandos de shell dentro de un
directorio raíz controlado (root_dir). El entorno de ejecución de cada comando
queda aislado mediante el parámetro env, que reemplaza las variables de entorno
del proceso padre por las que se especifiquen explícitamente.

Con virtual_mode=True el agente razona sobre los comandos y su salida pero no
los ejecuta realmente; es útil para explorar qué haría el agente sin riesgo de
efectos secundarios en el sistema.

Casos de uso habituales:
- Automatizar tareas de sistema: listar procesos, comprobar puertos, inspeccionar logs.
- Ejecutar scripts y comandos de proyecto en un entorno controlado y reproducible.
- Explorar comandos de forma segura antes de ejecutarlos en modo real (virtual_mode=True).
- Restringir el PATH disponible para evitar que el agente use herramientas no autorizadas.

Patrón utilizado:

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        backend=LocalShellBackend(
            root_dir=".",
            virtual_mode=True,
            env={"PATH": "/usr/bin:/bin"},
        ),
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Muestra los ficheros del directorio actual"}]}
    )

Parámetros de LocalShellBackend:
    root_dir     -- directorio de trabajo desde el que se ejecutan los comandos.
                    Puede ser absoluto o relativo al CWD (por defecto «.»).
    virtual_mode -- si es True, los comandos no se ejecutan realmente; el agente
                    simula la salida en memoria. Si es False, los comandos se
                    ejecutan en el shell del sistema con efectos reales.
    env          -- diccionario de variables de entorno disponibles durante la
                    ejecución de cada comando. Reemplaza el entorno del proceso
                    padre; define solo lo estrictamente necesario para reducir
                    la superficie de ataque (p. ej. PATH mínimo).

Trazas de log:
    El módulo usa el logger «localshellbackend» con dos niveles útiles:
    - INFO  (por defecto): agente construido, query enviada, duración e invocación.
    - DEBUG: trazas internas de LangChain/deepagents (pasos del grafo, callbacks).

    Para activar trazas remotas en LangSmith, define en .env:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<tu clave>
        LANGCHAIN_PROJECT=<nombre del proyecto>   # opcional

Formas de ejecutar:

    # 1. Consulta por defecto en modo virtual: el agente simula listar los ficheros
    > python myexamples_backends/03_localshellbackend_backend.py

    # 2. Consultar información del sistema sin ejecutar nada realmente
    > python myexamples_backends/03_localshellbackend_backend.py \
          "¿Qué comandos usarías para ver los procesos en ejecución?"

    # 3. Ejecutar un comando real en el directorio raíz (virtual_mode=False)
    #    PRECAUCIÓN: los comandos tienen efecto real en el sistema
    > python myexamples_backends/03_localshellbackend_backend.py \
          --no-virtual \
          "Muestra el contenido del fichero requirements.txt con cat"

    # 4. Restringir el directorio de trabajo y el PATH disponible
    > python myexamples_backends/03_localshellbackend_backend.py \
          --root-dir ./examples \
          --path "/usr/bin:/bin" \
          "Lista los ficheros .py disponibles"

    # 5. Añadir variables de entorno personalizadas para el agente
    > python myexamples_backends/03_localshellbackend_backend.py \
          --env "APP_ENV=test,LOG_LEVEL=debug" \
          "Muestra las variables de entorno disponibles"

    # 6. Activar trazas DEBUG locales
    > python myexamples_backends/03_localshellbackend_backend.py \
          --log-level DEBUG \
          "¿Qué es LocalShellBackend?"

"""

import argparse
import logging
import sys
import time
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_utils.logger import configure_logging  # noqa: E402

logger = logging.getLogger("localshellbackend")

_DEFAULT_PATH = "/usr/bin:/bin"


def _parse_env(env_str: str) -> dict[str, str]:
    """Convierte una cadena «CLAVE=VALOR,CLAVE2=VALOR2» en un diccionario.

    Permite pasar variables de entorno adicionales desde la CLI sin necesidad
    de modificar el fichero .env. Las claves o valores vacíos se ignoran.

    Args:
        env_str: Cadena con pares clave=valor separados por comas.
                 Ejemplo: «APP_ENV=test,LOG_LEVEL=debug».

    Returns:
        Diccionario con las variables de entorno parseadas.
    """
    result: dict[str, str] = {}
    for pair in env_str.split(","):
        pair = pair.strip()
        if "=" in pair:
            key, _, value = pair.partition("=")
            if key.strip():
                result[key.strip()] = value.strip()
    return result


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos del ejemplo.

    Returns:
        Namespace con los campos:
            query      -- consulta enviada al agente como mensaje de usuario.
            root_dir   -- directorio de trabajo desde el que el agente ejecuta
                          los comandos (por defecto «.»).
            path       -- valor del PATH inyectado en el entorno del shell del
                          agente (por defecto «/usr/bin:/bin»).
            env        -- variables de entorno adicionales en formato
                          «CLAVE=VALOR,CLAVE2=VALOR2».
            no_virtual -- si está presente, desactiva virtual_mode; los comandos
                          del agente se ejecutan realmente en el sistema.
            log_level  -- nivel de trazas de log («DEBUG», «INFO», «WARNING»).
    """
    parser = argparse.ArgumentParser(
        description="Ejemplo de LocalShellBackend con entorno de shell controlado"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Muestra los ficheros del directorio actual ordenados por nombre",
        help="Consulta a enviar al agente",
    )
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Directorio de trabajo del agente (por defecto: «.»)",
    )
    parser.add_argument(
        "--path",
        default=_DEFAULT_PATH,
        help=f"Valor de PATH para el entorno del shell (por defecto: «{_DEFAULT_PATH}»)",
    )
    parser.add_argument(
        "--env",
        default="",
        help="Variables de entorno adicionales en formato «CLAVE=VALOR,CLAVE2=VALOR2»",
    )
    parser.add_argument(
        "--no-virtual",
        action="store_true",
        help="Desactiva virtual_mode: los comandos se ejecutan realmente en el sistema",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Nivel de trazas de log (por defecto: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del ejemplo de LocalShellBackend.

    Flujo:
        1. Parsea los argumentos CLI (query, --root-dir, --path, --env,
           --no-virtual, --log-level).
        2. Configura el sistema de logging con el nivel indicado.
        3. Construye el entorno del shell combinando PATH y variables extra.
        4. Crea el agente con LocalShellBackend apuntando a root_dir.
        5. Invoca el agente registrando inicio, duración y resultado.
        6. Imprime el contenido del último mensaje de la respuesta.

    Con virtual_mode=True (por defecto) los comandos no se ejecutan realmente.
    Con --no-virtual los comandos tienen efecto real en el sistema: usar con
    precaución y solo en directorios y entornos de prueba.

    Termina con sys.exit si el agente lanza una excepción en tiempo de ejecución.
    """
    args = parse_args()
    configure_logging(logger, args.log_level)

    virtual_mode = not args.no_virtual
    mode_label = "virtual (sin ejecución real)" if virtual_mode else "real (ejecución en sistema)"

    shell_env: dict[str, str] = {"PATH": args.path}
    if args.env:
        shell_env.update(_parse_env(args.env))

    logger.info(
        "Construyendo agente con LocalShellBackend (root_dir=%s, virtual_mode=%s, PATH=%s)",
        args.root_dir, virtual_mode, args.path,
    )
    logger.debug("Entorno de shell completo: %s", shell_env)

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        backend=LocalShellBackend(
            root_dir=args.root_dir,
            virtual_mode=virtual_mode,
            env=shell_env,
        ),
    )

    logger.info("Modo de shell: %s", mode_label)
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
    print(f"[LocalShellBackend · {mode_label}] Resultado: {content}")


if __name__ == "__main__":
    main()
