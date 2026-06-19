"""
Ejemplo de uso de FilesystemBackend como backend del agente.

FilesystemBackend permite al agente leer y escribir ficheros dentro de un
directorio raíz controlado (root_dir). Con virtual_mode=True el agente opera
sobre una capa virtual en memoria: las escrituras no modifican el sistema de
ficheros real pero el agente puede razonar sobre ellas como si existieran.

Casos de uso habituales:
- Dar al agente acceso de lectura/escritura a ficheros del proyecto.
- Inspeccionar o generar código sin modificar el disco (virtual_mode=True).
- Combinar con CompositeBackend para mezclar acceso a ficheros y memoria.

Patrón utilizado:

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        backend=FilesystemBackend(root_dir=".", virtual_mode=True),
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Lista los ficheros del directorio raíz"}]}
    )

Parámetros de FilesystemBackend:
    root_dir     -- directorio raíz al que el agente tiene acceso.
                    Puede ser absoluto o relativo al CWD en el momento de la
                    invocación (por defecto «.», el directorio de trabajo actual).
    virtual_mode -- si es True, las operaciones de escritura no persisten en
                    disco; el agente trabaja sobre una capa en memoria.
                    Si es False, las escrituras son reales y permanentes.

Trazas de log:
    El módulo usa el logger «filesystembackend» con dos niveles útiles:
    - INFO  (por defecto): agente construido, query enviada, duración e invocación.
    - DEBUG: trazas internas de LangChain/deepagents (pasos del grafo, callbacks).

    Para activar trazas remotas en LangSmith, define en .env:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<tu clave>
        LANGCHAIN_PROJECT=<nombre del proyecto>   # opcional

Formas de ejecutar:

    # 1. Consulta por defecto: el agente lista los ficheros del directorio raíz
    > python myexamples_backends/02_filesystembackend_backend.py

    # 2. Consulta personalizada sobre el sistema de ficheros
    > python myexamples_backends/02_filesystembackend_backend.py \
          "Muestra el contenido del fichero requirements.txt"

    # 3. Cambiar el directorio raíz al que el agente tiene acceso
    > python myexamples_backends/02_filesystembackend_backend.py \
          --root-dir ./examples \
          "¿Qué ficheros hay disponibles?"

    # 4. Modo real (virtual_mode=False): las escrituras modifican el disco
    > python myexamples_backends/02_filesystembackend_backend.py \
          --no-virtual \
          "Crea un fichero llamado agente_test.txt con el texto 'Hola mundo'"

    # 5. Activar trazas DEBUG locales
    > python myexamples_backends/02_filesystembackend_backend.py \
          --log-level DEBUG \
          "Lista los ficheros del directorio raíz"

"""

import argparse
import logging
import sys
import time
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_utils.logger import configure_logging  # noqa: E402

logger = logging.getLogger("filesystembackend")


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos del ejemplo.

    Returns:
        Namespace con los campos:
            query      -- consulta enviada al agente como mensaje de usuario.
            root_dir   -- directorio raíz al que FilesystemBackend da acceso al
                          agente (por defecto «.», el directorio de trabajo actual).
            no_virtual -- si está presente, desactiva virtual_mode; las escrituras
                          del agente modificarán el disco real.
            log_level  -- nivel de trazas de log («DEBUG», «INFO», «WARNING»).
    """
    parser = argparse.ArgumentParser(
        description="Ejemplo de FilesystemBackend con acceso controlado al sistema de ficheros"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Lista los ficheros disponibles en el directorio raíz",
        help="Consulta a enviar al agente",
    )
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Directorio raíz al que el agente tiene acceso (por defecto: «.»)",
    )
    parser.add_argument(
        "--no-virtual",
        action="store_true",
        help="Desactiva virtual_mode: las escrituras modifican el disco real",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Nivel de trazas de log (por defecto: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada del ejemplo de FilesystemBackend.

    Flujo:
        1. Parsea los argumentos CLI (query, --root-dir, --no-virtual, --log-level).
        2. Configura el sistema de logging con el nivel indicado.
        3. Construye el agente con FilesystemBackend apuntando a root_dir.
        4. Invoca el agente registrando inicio, duración y resultado.
        5. Imprime el contenido del último mensaje de la respuesta.

    Con virtual_mode=True (por defecto) las escrituras del agente no persisten
    en disco; con --no-virtual las escrituras son reales y permanentes.

    Termina con sys.exit si el agente lanza una excepción en tiempo de ejecución.
    """
    args = parse_args()
    configure_logging(logger, args.log_level)

    virtual_mode = not args.no_virtual
    mode_label = "virtual (sin escritura en disco)" if virtual_mode else "real (escritura en disco)"

    logger.info("Construyendo agente con FilesystemBackend (root_dir=%s, virtual_mode=%s)", args.root_dir, virtual_mode)

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        backend=FilesystemBackend(root_dir=args.root_dir, virtual_mode=virtual_mode),
    )

    logger.info("Modo de ficheros: %s", mode_label)
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
    print(f"[FilesystemBackend · {mode_label}] Resultado: {content}")


if __name__ == "__main__":
    main()
