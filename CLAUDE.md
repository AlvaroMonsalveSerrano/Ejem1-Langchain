# CLAUDE.md

Este fichero proporciona orientación a Claude Code (claude.ai/code) cuando trabaja con el código de este repositorio.

## Configuración

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # luego editar con las API keys reales
```

## Comandos

```bash
# Ejecutar un ejemplo
python examples/01_basic_chain.py

# Ejecutar todos los ejemplos en orden
for f in examples/0*.py; do python "$f"; done

# Ejecutar todos los tests
pytest

# Ejecutar un test concreto
pytest tests/test_basic_chain.py::test_chain_returns_string -v
```

## Arquitectura

Sandbox de pruebas conceptuales para LangChain. Cada fichero numerado en `examples/` es un script autocontenido que demuestra un concepto. Los scripts están diseñados para ejecutarse directamente (guardia `__main__`) e imprimen el resultado por stdout.

**Patrón central**: composición LCEL con pipe — `prompt | model | parser`. Todos los ejemplos se construyen sobre este patrón.

| Fichero | Concepto |
|---------|----------|
| `01_basic_chain.py` | Cadena básica: `ChatPromptTemplate | ChatOpenAI | StrOutputParser` |
| `02_prompt_templates.py` | Variantes de prompts: multi-variable, few-shot |
| `03_output_parsers.py` | `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser` |

Los tests se encuentran en `tests/` y usan `FakeListLLM` para evitar llamadas reales a la API.

## Convenciones

- Todos los ejemplos cargan las credenciales mediante `python-dotenv` (`load_dotenv()` al inicio).
- Los nuevos ejemplos deben seguir el esquema `NN_nombre_concepto.py` y ubicarse en `examples/`.
- Usar `gpt-4o-mini` como modelo por defecto para iterar con coste bajo; cambiar a un modelo más capaz solo cuando el concepto lo requiera.
- Los modelos Pydantic para salida estructurada se definen en el mismo fichero del ejemplo que los usa (no se necesita un módulo de modelos compartido a esta escala).
