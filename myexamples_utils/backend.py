
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langgraph.store.memory import InMemoryStore

def make_backend(runtime):
    """Construye el backend compuesto que gestiona el estado y la memoria del agente.

    Combina dos backends bajo un único CompositeBackend:
    - StateBackend  (ruta por defecto): persiste el estado conversacional del grafo
      (mensajes, campos de ResearchState, etc.) durante la ejecución del agente.
    - StoreBackend  (ruta «/memories/»): gestiona la memoria a largo plazo mediante
      InMemoryStore. La ruta «/memories/» es una ruta virtual — no corresponde a
      ningún directorio en el sistema de ficheros. StoreBackend la intercepta y
      almacena las entradas como claves en InMemoryStore (en memoria del proceso).
      El agente puede escribir en rutas como «/memories/user_preferences.txt» y
      StoreBackend las persiste como claves del store mientras viva el proceso.

    Args:
        runtime: Instancia de runtime de deepagents que proporciona acceso al
                 estado del grafo y al store subyacente.

    Returns:
        CompositeBackend configurado con StateBackend como ruta por defecto y
        StoreBackend montado en «/memories/».
    """
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={"/memories/": StoreBackend(runtime)},
    )