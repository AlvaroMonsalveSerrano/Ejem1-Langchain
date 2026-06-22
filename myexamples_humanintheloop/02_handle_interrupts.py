"""
Ejemplo de gestión de interrupciones con la API v2 de deepagents.

Este módulo demuestra el mismo patrón Human-in-the-Loop que 01_basic.py pero
usando la API ``version="v2"`` de ``agent.invoke()``. La diferencia principal
es que v2 devuelve un objeto tipado en lugar de un diccionario plano, lo que
hace el código más legible y robusto.

Diferencias clave respecto a 01_basic.py (API v1):

    v1 (01_basic.py)                     v2 (este módulo)
    ─────────────────────────────────    ──────────────────────────────────────
    result["__interrupt__"]              result.interrupts
    result["__interrupt__"][0].value     result.interrupts[0].value
    result["messages"][-1].content       result.value["messages"][-1].content
    uuid.uuid4()                         uuid7()  (time-ordered)
    invoke(...) sin version              invoke(..., version="v2")

uuid7 frente a uuid4:
    ``uuid7`` genera identificadores ordenados por tiempo (los primeros bytes
    codifican el timestamp en milisegundos). Esto mejora el rendimiento en
    bases de datos que indexan por el thread_id (B-tree, índices de LangSmith)
    al reducir la fragmentación de índices. uuid4 es aleatorio puro.

Decisión «reject» con campo «message»:
    En v2 la decisión de rechazo admite un campo opcional «message» que se
    pasa al LLM como contexto para que explique al usuario por qué no ejecutó
    la herramienta y evite reintentos innecesarios:

        {"type": "reject", "message": "Borrado cancelado por el usuario."}

    Sin «message» el LLM recibe solo el rechazo sin explicación.

Patrón utilizado:

    from langchain_core.utils.uuid import uuid7
    from langgraph.types import Command

    config = {"configurable": {"thread_id": str(uuid7())}}

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Elimina temp.txt"}]},
        config=config,
        version="v2",
    )

    if result.interrupts:
        interrupt_value = result.interrupts[0].value
        action_requests = interrupt_value["action_requests"]
        review_configs  = interrupt_value["review_configs"]

        config_map = {cfg["action_name"]: cfg for cfg in review_configs}
        for action in action_requests:
            rc = config_map[action["name"]]
            print(action["name"], action["args"], rc["allowed_decisions"])

        decisions = [{"type": "reject", "message": "Cancelado por el usuario."}]

        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,
            version="v2",
        )

    print(result.value["messages"][-1].content)

Flujo de ejecución con interrupción (v2):
    1. ``agent.invoke(..., version="v2")`` devuelve un objeto con:
           .interrupts  -- lista de Interrupt pendientes (vacía si no hay).
           .value       -- estado del grafo (solo disponible cuando no hay
                           interrupciones pendientes).
    2. Si ``result.interrupts`` no está vacío, se extrae el payload de
       ``result.interrupts[0].value``:
           {
             "action_requests": [{"name": "...", "args": {...}}],
             "review_configs":  [{"action_name": "...", "allowed_decisions": [...]}]
           }
    3. El usuario decide por cada acción. El payload de reanudación es:
           {"decisions": [{"type": "<decisión>"}]}
       Campos adicionales según la decisión:
           reject  → añade "message": "..."          (motivo para el LLM, opcional)
           edit    → añade "edited_action": {"name": "<herramienta>", "args": {...}}
           respond → añade "response": "..."
    4. Se reanuda con ``Command(resume=payload)`` usando el mismo config.
    5. El ciclo continúa hasta que ``result.interrupts`` esté vacío.
    6. El resultado final se lee en ``result.value["messages"][-1].content``.

Trazas de log:
    El módulo usa el logger «handleinterrupts» con dos niveles útiles:
    - INFO  (por defecto): herramientas registradas, query enviada,
            interrupciones detectadas, decisión del usuario y resultado.
    - DEBUG: payload completo de cada interrupción y trazas internas
             de LangChain/deepagents.

    Para activar trazas remotas en LangSmith, define en .env:
        LANGCHAIN_TRACING_V2=true
        LANGCHAIN_API_KEY=<tu clave>
        LANGCHAIN_PROJECT=<nombre del proyecto>   # opcional

Formas de ejecutar:

    # 1. Consulta por defecto: remove_file interrumpe → approve
    > python myexamples_humanintheloop/02_handle_interrupts.py

    # 2. Leer un fichero (sin interrupción: fetch_file tiene interrupt_on=False)
    > python myexamples_humanintheloop/02_handle_interrupts.py \
          "Lee el contenido del fichero /tmp/datos.txt"

    # 3. Borrar un fichero → reject con motivo (el LLM recibe el mensaje)
    > python myexamples_humanintheloop/02_handle_interrupts.py \
          "Elimina el fichero /tmp/temporal.log"
    #    → introducir: reject
    #    → motivo: Fichero en uso, no se puede borrar ahora.

    # 4. Borrar un fichero → edit para cambiar la ruta antes de ejecutar
    > python myexamples_humanintheloop/02_handle_interrupts.py \
          "Elimina el fichero /tmp/temporal.log"
    #    → introducir: edit
    #    → JSON: {"path": "/tmp/temporal_backup.log"}

    # 5. Enviar correo → approve (solo approve/reject disponibles)
    > python myexamples_humanintheloop/02_handle_interrupts.py \
          "Envía un correo a dev@example.com con asunto 'Reporte' y cuerpo 'Todo OK'"

    # 6. Activar trazas DEBUG para ver el payload completo de interrupción
    > python myexamples_humanintheloop/02_handle_interrupts.py \
          --log-level DEBUG \
          "Elimina el fichero /tmp/ejemplo.log"

"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from myexamples_utils.logger import configure_logging  # noqa: E402

logger = logging.getLogger("handleinterrupts")


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
    logger.debug("notify_email: cuerpo del mensaje: %s", body)
    return f"Correo enviado correctamente a '{to}'."


# ---------------------------------------------------------------------------
# Gestión interactiva de interrupciones (API v2)
# ---------------------------------------------------------------------------


def handle_interrupt(interrupt) -> dict:
    """Muestra la interrupción al usuario y recoge su decisión (API v2).

    En v2 el objeto Interrupt se obtiene de ``result.interrupts[0]``. Su
    atributo ``.value`` tiene la misma estructura que en v1:
        {
          "action_requests": [{"name": "...", "args": {...}, "description": "..."}],
          "review_configs":  [{"action_name": "...", "allowed_decisions": [...]}]
        }

    La diferencia respecto a v1 está en el payload de retorno: en v2 no es
    necesario incluir «action_name» en cada entrada de decisions porque el
    framework las empareja por posición con action_requests. El campo
    «message» es exclusivo de v2 y solo aplica a la decisión «reject».

    Args:
        interrupt: objeto Interrupt de LangGraph (``result.interrupts[0]``).

    Returns:
        Diccionario con el formato v2 para reanudar:
            {"decisions": [{"type": "approve"}, ...]}
        Campos adicionales según la decisión:
            reject  → añade "message": "..."          (motivo opcional para el LLM)
            edit    → añade "edited_action": {"name": "<herramienta>", "args": {...}}
            respond → añade "message": "..."         (respuesta personalizada para el LLM)
    """
    value = getattr(interrupt, "value", interrupt)
    logger.debug("Payload de interrupción (v2): %s", value)

    if not isinstance(value, dict):
        logger.warning("Formato de interrupción inesperado: %s", value)
        return {"decisions": []}

    action_requests = value.get("action_requests", [])
    review_configs = value.get("review_configs", [])

    config_map = {rc["action_name"]: rc for rc in review_configs}

    decisions = []
    for action in action_requests:
        tool_name = action.get("name", "desconocida")
        tool_args = action.get("args", {})
        rc = config_map.get(tool_name, {})
        allowed = rc.get("allowed_decisions", ["approve", "edit", "reject", "respond"])

        logger.info("Herramienta pendiente  : %s", tool_name)
        logger.info("Argumentos propuestos  : %s", json.dumps(tool_args, ensure_ascii=False))
        logger.info("Decisiones disponibles : %s", ", ".join(allowed))

        while True:
            decision = input(f"  Tu decisión para «{tool_name}» [{'/'.join(allowed)}]: ").strip().lower()
            if decision in allowed:
                break
            logger.warning("Opción no válida. Elige entre: %s", ", ".join(allowed))

        logger.info("Decisión del usuario para «%s»: %s", tool_name, decision)

        entry: dict = {"type": decision}
        if decision == "reject":
            msg = input("  Motivo del rechazo (opcional, Enter para omitir): ").strip()
            if msg:
                entry["message"] = msg
        elif decision == "edit":
            raw = input("  Nuevos argumentos (JSON): ").strip()
            try:
                entry["edited_action"] = {"name": tool_name, "args": json.loads(raw)}
            except json.JSONDecodeError:
                logger.warning("JSON inválido; se mantienen los argumentos originales.")
        elif decision == "respond":
            entry["message"] = input("  Respuesta personalizada: ").strip()

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
        description="Gestión de interrupciones HITL con API v2 de deepagents"
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
    """Punto de entrada del ejemplo de gestión de interrupciones (API v2).

    Flujo:
        1. Parsea los argumentos CLI (query, --log-level).
        2. Configura el sistema de logging.
        3. Construye el agente con interrupt_on y MemorySaver como checkpointer.
        4. Genera un thread_id con uuid7 (time-ordered) y construye el config.
        5. Invoca el agente con version="v2".
        6. Itera mientras result.interrupts no esté vacío:
           - Extrae el payload de result.interrupts[0].
           - Recoge la decisión del usuario.
           - Reanuda con Command(resume=payload) y version="v2".
        7. Lee el resultado final en result.value["messages"][-1].content.

    Termina con sys.exit si el agente lanza una excepción en tiempo de ejecución.
    """
    args = parse_args()
    configure_logging(logger, args.log_level)

    logger.info("Construyendo agente con human-in-the-loop (API v2)")

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

    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}

    logger.info("Thread ID (uuid7): %s", thread_id)
    logger.info("Herramientas: remove_file (interrupt=True), fetch_file (interrupt=False), "
                "notify_email (interrupt=approve/reject)")
    logger.info("Query: %s", args.query)

    t0 = time.perf_counter()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": args.query}]},
            config=config,
            version="v2",
        )

        # Ciclo de gestión de interrupciones con API v2
        while result.interrupts:
            logger.info("%d interrupción/es pendiente/s", len(result.interrupts))
            resume_payload = handle_interrupt(result.interrupts[0])
            result = agent.invoke(
                Command(resume=resume_payload),
                config=config,
                version="v2",
            )

    except Exception as exc:
        logger.error("Error al invocar el agente: %s", exc)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    logger.info("Invocación completada en %.2f s", elapsed)

    messages = result.value["messages"]

    # Resultados reales de las herramientas ejecutadas (fuente de verdad)
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    for tm in tool_msgs:
        logger.info("Resultado de herramienta: %s", tm.content)

    ai_content = messages[-1].content
    logger.debug("Respuesta del agente: %s", ai_content)

    # Si hubo herramientas ejecutadas, mostrar su resultado real además
    # del resumen del agente, para que las ediciones de ruta sean visibles.
    if tool_msgs:
        print(f"[HandleInterrupts v2] Ejecución real  : {tool_msgs[-1].content}")
    print(f"[HandleInterrupts v2] Respuesta agente: {ai_content}")


if __name__ == "__main__":
    main()
