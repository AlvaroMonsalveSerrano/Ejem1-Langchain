"""Utilidades de logging compartidas para los ejemplos de myexamples_backends."""

import logging


def configure_logging(logger: logging.Logger, level_name: str) -> None:
    """Configura el sistema de logging local para un ejemplo.

    Establece el formato y nivel del logger proporcionado. Si el nivel es DEBUG,
    activa también los loggers internos de langchain y deepagents para exponer
    los pasos del grafo y los callbacks de LangChain.

    Las trazas remotas en LangSmith se activan por separado mediante las
    variables de entorno LANGCHAIN_TRACING_V2 y LANGCHAIN_API_KEY definidas
    en el fichero .env.

    Args:
        logger:     Logger del módulo llamante, obtenido con
                    logging.getLogger("<nombre>"). Su nivel se ajusta al valor
                    de level_name; los loggers de langchain y deepagents solo
                    se modifican si level_name es «DEBUG».
        level_name: Nombre del nivel de log en mayúsculas («DEBUG», «INFO»,
                    «WARNING»). Se mapea a las constantes del módulo logging
                    (logging.DEBUG, logging.INFO, etc.).
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.WARNING,
    )
    logger.setLevel(level)
    if level <= logging.DEBUG:
        logging.getLogger("langchain").setLevel(logging.DEBUG)
        logging.getLogger("deepagents").setLevel(logging.DEBUG)
