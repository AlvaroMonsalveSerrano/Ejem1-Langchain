"""
Ejemplo de uso de FilesystemPermission para controlar el acceso del agente.

FilesystemPermission permite definir reglas de acceso que el agente debe respetar
al operar sobre el sistema de ficheros. Cada regla combina tres dimensiones:

    operations -- qué operaciones afecta: "read", "write" o ambas.
    paths      -- rutas o patrones glob a los que aplica la regla (p. ej. "/**").
    mode       -- "allow" (permitir) o "deny" (denegar).

Las reglas se evalúan en orden; la primera que coincide decide el resultado.

Patrón utilizado (agente de solo lectura):

    from deepagents import FilesystemPermission, create_deep_agent
    from deepagents.backends import FilesystemBackend

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        backend=FilesystemBackend(root_dir=".", virtual_mode=True),
        permissions=[
            FilesystemPermission(
                operations=["write"],
                paths=["/**"],
                mode="deny",
            ),
        ],
    )

Casos de uso habituales:
- Agente de auditoría: solo lectura, sin capacidad de modificar ficheros.
- Agente de generación acotada: escritura permitida únicamente en un subdirectorio.
- Agente de sandbox: acceso total a /tmp pero nada fuera de esa ruta.

Trazas de log:
    El módulo usa el logger «permissions» con dos niveles útiles:
    - INFO  (por defecto): configuración de permisos, query enviada y duración.
    - DEBUG: trazas internas de LangChain/deepagents (pasos del grafo, callbacks).

    Para activar trazas remotas en LangSmith, define en .env:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<tu clave>
        LANGCHAIN_PROJECT=<nombre del proyecto>   # opcional

Formas de ejecutar:

    # 1. Comportamiento por defecto: agente de solo lectura (deniega escrituras)
    > python myexamples_permissions/01_basic_usage_permissions.py

    # 2. Consulta de lectura explícita — el agente puede responder sin problema
    > python myexamples_permissions/01_basic_usage_permissions.py \
          "Muestra el contenido del fichero requirements.txt"

    # 3. Intento de escritura con permiso denegado — el agente debe rechazarlo
    > python myexamples_permissions/01_basic_usage_permissions.py \
          "Crea un fichero llamado test_permiso.txt con el texto 'Hola'"

    # 4. Mismo intento de escritura pero con escrituras permitidas (--allow-writes)
    > python myexamples_permissions/01_basic_usage_permissions.py \
          --allow-writes \
          "Crea un fichero llamado test_permiso.txt con el texto 'Hola'"

    # 5. Acotar la denegación a un subdirectorio concreto
    > python myexamples_permissions/01_basic_usage_permissions.py \
          --deny-path "/examples/**" \
          "Modifica el fichero examples/01_basic_chain.py añadiendo un comentario"

    # 6. Cambiar el directorio raíz al que el agente tiene acceso
    > python myexamples_permissions/01_basic_usage_permissions.py \
          --root-dir ./examples \
          "Lista los ficheros disponibles"

    # 7. Activar trazas DEBUG locales
    > python myexamples_permissions/01_basic_usage_permissions.py \
          --log-level DEBUG \
          "Lista los ficheros del directorio raíz"

"""

import argparse
import logging
import sys
import time
from pathlib import Path

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_utils.logger import configure_logging  # noqa: E402

logger = logging.getLogger("permissions")


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos del ejemplo.

    Returns:
        Namespace con los campos:
            query       -- consulta enviada al agente como mensaje de usuario.
            root_dir    -- directorio raíz al que FilesystemBackend da acceso.
            allow_writes -- si está presente, no se añade la regla de denegación
                           de escrituras; útil para contrastar el comportamiento.
            deny_path   -- patrón glob al que se aplica la denegación de escritura
                           (por defecto «/**», es decir, todo el árbol).
            log_level   -- nivel de trazas de log («DEBUG», «INFO», «WARNING»).
    """
    parser = argparse.ArgumentParser(
        description="Ejemplo de FilesystemPermission: control de acceso del agente al sistema de ficheros"
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
        "--allow-writes",
        action="store_true",
        help="Permite escrituras al agente (desactiva la regla deny-write por defecto)",
    )
    parser.add_argument(
        "--deny-path",
        default="/**",
        help="Patrón glob al que se aplica la denegación de escritura (por defecto: «/**»)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Nivel de trazas de log (por defecto: INFO)",
    )
    return parser.parse_args()


def build_permissions(allow_writes: bool, deny_path: str) -> list:
    """Construye la lista de FilesystemPermission según los argumentos.

    Con allow_writes=False (por defecto) se añade una regla que deniega
    cualquier operación de escritura sobre deny_path, convirtiendo al agente
    en un agente de solo lectura.

    Con allow_writes=True la lista de permisos queda vacía y el agente opera
    con acceso completo (limitado únicamente por root_dir del backend).

    Args:
        allow_writes: si True no se añade ninguna restricción de escritura.
        deny_path:    patrón glob al que se aplica la denegación de escritura.

    Returns:
        Lista de FilesystemPermission (puede estar vacía).
    """
    if allow_writes:
        return []

    return [
        FilesystemPermission(
            operations=["write"],
            paths=[deny_path],
            mode="deny",
        )
    ]


def main() -> None:
    """Punto de entrada del ejemplo de FilesystemPermission.

    Flujo:
        1. Parsea los argumentos CLI.
        2. Configura el sistema de logging.
        3. Construye la lista de permisos según --allow-writes y --deny-path.
        4. Crea el agente con FilesystemBackend y los permisos definidos.
        5. Invoca el agente con la query indicada registrando duración y resultado.

    El agente siempre opera sobre una capa virtual (virtual_mode=True) para que
    las escrituras permitidas no modifiquen el disco durante las pruebas.

    Termina con sys.exit si el agente lanza una excepción en tiempo de ejecución.
    """
    args = parse_args()
    configure_logging(logger, args.log_level)

    permissions = build_permissions(args.allow_writes, args.deny_path)

    if permissions:
        logger.info(
            "Permiso activo: deny write en '%s' → agente de solo lectura",
            args.deny_path,
        )
    else:
        logger.info("Sin restricciones de permisos → agente con acceso completo")

    logger.info("Construyendo agente (root_dir=%s, virtual_mode=True)", args.root_dir)

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        backend=FilesystemBackend(root_dir=args.root_dir, virtual_mode=True),
        permissions=permissions,
    )

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

    perm_label = "deny-write" if permissions else "allow-all"
    content = result["messages"][-1].content
    logger.debug("Respuesta completa: %s", content)
    print(f"[FilesystemPermission · {perm_label}] Resultado: {content}")


if __name__ == "__main__":
    main()
