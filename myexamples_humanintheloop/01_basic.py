"""
Ejemplo básico de Human-in-the-Loop con deepagents.

Human-in-the-loop (HITL) permite que un humano revise y apruebe (o rechace)
las llamadas a herramientas del agente antes de que se ejecuten. Es especialmente
útil para operaciones con efectos secundarios irreversibles —borrar ficheros,
enviar correos— donde un error del agente podría ser costoso.

El parámetro ``interrupt_on`` define qué herramientas pausan la ejecución y
qué decisiones puede tomar el humano para cada una:

    True
        Habilita interrupciones con el conjunto completo de decisiones:
        approve, edit, reject, respond.
    False
        Sin interrupción: la herramienta se ejecuta directamente.
    {"allowed_decisions": [...]}
        Limita las decisiones disponibles para esa herramienta.

Decisiones disponibles:
    approve   -- ejecutar la herramienta con los argumentos propuestos.
    edit      -- modificar los argumentos antes de ejecutar la herramienta.
    reject    -- cancelar la llamada sin ejecutar la herramienta.
    respond   -- proporcionar un resultado personalizado sin ejecutar la herramienta.

El checkpointer es OBLIGATORIO para HITL. LangGraph lo usa internamente para
guardar el estado del grafo en el punto de interrupción y restaurarlo cuando
el humano proporciona su decisión. Sin checkpointer el agente no puede pausar
ni reanudar la ejecución.

Herramientas en este ejemplo y su configuración de interrupción:
    remove_file   -- elimina un fichero del sistema.
                     interrupt_on=True: todas las decisiones, porque el borrado
                     es irreversible.
    fetch_file    -- lee el contenido de un fichero.
                     interrupt_on=False: sin interrupción; es operación de solo
                     lectura y no tiene efectos secundarios.
    notify_email  -- envía un correo electrónico.
                     interrupt_on={"allowed_decisions": ["approve", "reject"]}:
                     solo aprobar o rechazar; no se permite editar el mensaje
                     parcialmente para mantener la coherencia del contenido.

Patrón utilizado:

    from langchain.tools import tool
    from deepagents import create_deep_agent
    from langgraph.checkpoint.memory import MemorySaver

    @tool
    def remove_file(path: str) -> str:
        \"\"\"Elimina un fichero del sistema de ficheros.\"\"\"
        return f"Deleted {path}"

    @tool
    def fetch_file(path: str) -> str:
        \"\"\"Lee el contenido de un fichero del sistema de ficheros.\"\"\"
        return f"Contents of {path}"

    @tool
    def notify_email(to: str, subject: str, body: str) -> str:
        \"\"\"Envía un correo electrónico.\"\"\"
        return f"Sent email to {to}"

    checkpointer = MemorySaver()

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[remove_file, fetch_file, notify_email],
        interrupt_on={
            "remove_file": True,
            "fetch_file": False,
            "notify_email": {"allowed_decisions": ["approve", "reject"]},
        },
        checkpointer=checkpointer,  # Obligatorio para HITL
    )

Flujo de ejecución con interrupción:
    1. El agente recibe el mensaje del usuario y decide llamar a una herramienta.
    2. Si esa herramienta tiene interrupt_on activo, la ejecución se pausa y
       el estado del grafo se guarda en el checkpointer.
    3. deepagents publica un objeto Interrupt cuyo .value tiene esta estructura:
           {
             "action_requests": [{"name": "...", "args": {...}, "description": "..."}],
             "review_configs":  [{"action_name": "...", "allowed_decisions": [...]}]
           }
    4. El sistema muestra al usuario la herramienta pendiente y sus argumentos,
       y le solicita una decisión (approve / edit / reject / respond).
    5. La ejecución se reanuda con Command(resume=payload) donde payload es:
           {"decisions": [{"action_name": "...", "type": "<decisión>"}]}
       Con campos adicionales según la decisión:
           edit    → añade "args": {...}
           respond → añade "response": "..."

Trazas de log:
    El módulo usa el logger «humanintheloop» con dos niveles útiles:
    - INFO  (por defecto): herramientas registradas, query enviada, interrupciones
            detectadas, decisión del usuario y resultado final.
    - DEBUG: trazas internas de LangChain/deepagents (pasos del grafo,
            callbacks, estado del checkpointer).

    Para activar trazas remotas en LangSmith, define en .env:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<tu clave>
        LANGCHAIN_PROJECT=<nombre del proyecto>   # opcional

Formas de ejecutar:

    # 1. Consulta por defecto: el agente intenta borrar un fichero (interrumpe)
    > python myexamples_humanintheloop/01_basic.py

    # 2. Leer un fichero (sin interrupción: fetch_file tiene interrupt_on=False)
    > python myexamples_humanintheloop/01_basic.py \
          "Lee el contenido del fichero /tmp/datos.txt"

    # 3. Borrar un fichero (interrumpe con todas las decisiones: approve/edit/reject/respond)
    > python myexamples_humanintheloop/01_basic.py \
          "Elimina el fichero /tmp/temporal.log"

    # 4. Enviar un correo (interrumpe con solo approve/reject, sin edit)
    > python myexamples_humanintheloop/01_basic.py \
          "Envía un correo a dev@example.com con asunto 'Reporte' y cuerpo 'Todo OK'"

    # 5. Activar trazas DEBUG para ver los pasos del grafo y el checkpointer
    > python myexamples_humanintheloop/01_basic.py \
          --log-level DEBUG \
          "Borra el fichero /var/log/app.log"

    # 6. Combinar herramientas: leer y luego notificar (solo notify_email interrumpe).
    #    La query debe nombrar explícitamente las herramientas para que el LLM
    #    las invoque en orden sin decidir por su cuenta si el fichero existe.
    > python myexamples_humanintheloop/01_basic.py \
          "Usa la herramienta fetch_file para leer /tmp/report.txt y después usa notify_email para enviar su contenido a admin@example.com"

"""

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_utils.logger import configure_logging  # noqa: E402

logger = logging.getLogger("humanintheloop")


# ---------------------------------------------------------------------------
# Herramientas del agente (simuladas para el ejemplo)
# ---------------------------------------------------------------------------


@tool
def remove_file(path: str) -> str:
    """Elimina un fichero del sistema de ficheros."""
    logger.info("remove_file: eliminando fichero '%s'", path)
    return f"Fichero '{path}' eliminado correctamente."


@tool
def fetch_file(path: str) -> str:
    """Lee el contenido de un fichero del sistema de ficheros."""
    logger.info("fetch_file: leyendo fichero '%s'", path)
    return (
        f"Contenido del fichero '{path}':\n"
        "---\n"
        "Informe de estado del sistema - 2026-06-22\n"
        "Errores detectados: 0\n"
        "Advertencias: 2 (uso de disco >80 %, certificado caduca en 15 días)\n"
        "Última comprobación: 08:00 UTC\n"
        "---"
    )


@tool
def notify_email(to: str, subject: str, body: str) -> str:
    """Envía un correo electrónico."""
    logger.info("notify_email: enviando correo a '%s' con asunto '%s'", to, subject)
    return f"Correo enviado correctamente a '{to}'."


# ---------------------------------------------------------------------------
# Gestión interactiva de interrupciones
# ---------------------------------------------------------------------------


def handle_interrupt(interrupt) -> dict:
    """Muestra la interrupción al usuario y recoge su decisión.

    deepagents entrega el payload con esta estructura:
        {
          "action_requests": [{"name": "...", "args": {...}, "description": "..."}],
          "review_configs":  [{"action_name": "...", "allowed_decisions": [...]}]
        }

    La función itera sobre cada action_request, muestra al usuario la herramienta
    y sus argumentos, y recoge una decisión por cada una.

    Args:
        interrupt: objeto Interrupt de LangGraph. Su atributo .value contiene
                   el payload descrito arriba.

    Returns:
        Diccionario con el formato que espera deepagents para reanudar:
            {"decisions": [{"action_name": "...", "type": "approve"}, ...]}
        Con campos adicionales según la decisión:
            edit    → añade "args": {...}
            respond → añade "response": "..."
    """
    value = getattr(interrupt, "value", interrupt)
    logger.debug("Payload de interrupción: %s", value)

    if not isinstance(value, dict):
        logger.warning("Formato de interrupción inesperado: %s", value)
        return {"decisions": []}

    action_requests = value.get("action_requests", [])
    review_configs = value.get("review_configs", [])

    allowed_map = {
        rc["action_name"]: rc.get("allowed_decisions", ["approve", "edit", "reject", "respond"])
        for rc in review_configs
    }

    decisions = []
    for action in action_requests:
        tool_name = action.get("name", "desconocida")
        tool_args = action.get("args", {})
        allowed = allowed_map.get(tool_name, ["approve", "edit", "reject", "respond"])

        logger.info("Herramienta pendiente  : %s", tool_name)
        logger.info("Argumentos propuestos  : %s", json.dumps(tool_args, ensure_ascii=False))
        logger.info("Decisiones disponibles : %s", ", ".join(allowed))

        while True:
            decision = input(f"  Tu decisión para «{tool_name}» [{'/'.join(allowed)}]: ").strip().lower()
            if decision in allowed:
                break
            logger.warning("Opción no válida. Elige entre: %s", ", ".join(allowed))

        logger.info("Decisión del usuario para «%s»: %s", tool_name, decision)

        entry: dict = {"action_name": tool_name, "type": decision}
        if decision == "edit":
            raw = input("  Nuevos argumentos (JSON): ").strip()
            try:
                entry["args"] = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("JSON inválido; se mantienen los argumentos originales.")
        elif decision == "respond":
            entry["response"] = input("  Respuesta personalizada: ").strip()

        decisions.append(entry)

    return {"decisions": decisions}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos.

    Returns:
        Namespace con los campos:
            query     -- consulta enviada al agente como mensaje de usuario.
            log_level -- nivel de trazas de log («DEBUG», «INFO», «WARNING»).
    """
    parser = argparse.ArgumentParser(
        description="Ejemplo básico de Human-in-the-Loop con deepagents"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Elimina el fichero /tmp/ejemplo.log",
        help="Consulta a enviar al agente",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Nivel de trazas de log (por defecto: INFO)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------


def main() -> None:
    """Punto de entrada del ejemplo de Human-in-the-Loop.

    Flujo:
        1. Parsea los argumentos CLI (query, --log-level).
        2. Configura el sistema de logging.
        3. Construye el agente con interrupt_on y MemorySaver como checkpointer.
        4. Invoca el agente dentro de un ciclo que gestiona interrupciones:
           - Si el resultado contiene __interrupt__, muestra la interrupción
             al usuario, recoge su decisión y reanuda la ejecución.
           - El ciclo continúa hasta que no queden interrupciones pendientes.
        5. Imprime el contenido del último mensaje de la respuesta.

    Cada ejecución genera un thread_id único para que el checkpointer aísle
    el estado de sesiones concurrentes.

    Termina con sys.exit si el agente lanza una excepción en tiempo de ejecución.
    """
    args = parse_args()
    configure_logging(logger, args.log_level)

    logger.info("Construyendo agente con human-in-the-loop")

    checkpointer = MemorySaver()

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        tools=[remove_file, fetch_file, notify_email],
        interrupt_on={
            "remove_file": True,
            "fetch_file": False,
            "notify_email": {"allowed_decisions": ["approve", "reject"]},
        },
        checkpointer=checkpointer,
    )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    logger.info("Thread ID : %s", thread_id)
    logger.info("Herramientas: remove_file (interrupt=True), fetch_file (interrupt=False), "
                "notify_email (interrupt=approve/reject)")
    logger.info("Query: %s", args.query)

    t0 = time.perf_counter()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": args.query}]},
            config=config,
        )

        # Ciclo de gestión de interrupciones
        while "__interrupt__" in result:
            interrupts = result["__interrupt__"]
            logger.info("%d interrupción/es pendiente/s", len(interrupts))

            # deepagents agrupa todos los action_requests en el primer Interrupt
            resume_payload = handle_interrupt(interrupts[0])
            result = agent.invoke(Command(resume=resume_payload), config=config)

    except Exception as exc:
        logger.error("Error al invocar el agente: %s", exc)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    logger.info("Invocación completada en %.2f s", elapsed)

    content = result["messages"][-1].content
    logger.debug("Respuesta completa: %s", content)
    print(f"[HumanInTheLoop] Resultado: {content}")


if __name__ == "__main__":
    main()
