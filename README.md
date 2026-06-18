# Ejem1-LangChain

Sandbox de pruebas conceptuales con la librería [LangChain](https://python.langchain.com/). Cada script en `examples/` es autocontenido y demuestra un concepto concreto del ecosistema LangChain, siguiendo el patrón LCEL (`prompt | model | parser`).

---

## Instalación del entorno

**Requisitos:** Python 3.10+

```bash
# 1. Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar credenciales
cp .env.example .env
# Editar .env y añadir las API keys necesarias:
#   OPENAI_API_KEY=sk-...
#   ANTHROPIC_API_KEY=sk-ant-...
```

---

## Ejemplos disponibles

| Archivo | Concepto |
|---------|----------|
| `examples/01_basic_chain.py` | Cadena básica: `ChatPromptTemplate | ChatOpenAI | StrOutputParser`. Punto de entrada mínimo para entender LCEL. |
| `examples/02_prompt_templates.py` | Variantes de `ChatPromptTemplate`: múltiples variables y prompts few-shot con `FewShotChatMessagePromptTemplate`. |
| `examples/03_output_parsers.py` | Los tres parsers principales: `StrOutputParser` (texto libre), `JsonOutputParser` (diccionario) y `PydanticOutputParser` (objeto tipado con validación). |

---

## Cómo ejecutar los ejemplos

```bash
# Activar el entorno virtual si no está activo
source .venv/bin/activate

# Ejecutar un ejemplo concreto
python examples/01_basic_chain.py

# Ejecutar todos los ejemplos en orden
for f in examples/0*.py; do echo "=== $f ==="; python "$f"; done
```

> Todos los ejemplos imprimen su resultado por stdout. Asegúrate de que `.env` está configurado antes de ejecutarlos.

---

## Tests

Los tests usan `pytest` y un LLM simulado (`FakeListLLM`) para no requerir API keys ni realizar llamadas reales.

```bash
# Ejecutar todos los tests
pytest

# Ejecutar un fichero de tests concreto
pytest tests/test_basic_chain.py

# Ejecutar un test específico por nombre
pytest tests/test_basic_chain.py::test_chain_returns_string

# Ver output detallado
pytest -v
```

Los tests se encuentran en el directorio `tests/` y siguen la convención `test_<nombre_del_ejemplo>.py`.
