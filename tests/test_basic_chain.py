"""
Tests para examples/01_basic_chain.py.
Usa un FakeListLLM para evitar llamadas reales a la API.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.fake import FakeListLLM


def build_chain(responses: list[str]):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil que responde en español."),
        ("human", "{pregunta}"),
    ])
    model = FakeListLLM(responses=responses)
    return prompt | model | StrOutputParser()


def test_chain_returns_string():
    chain = build_chain(["LangChain es un framework para construir apps con LLMs."])
    result = chain.invoke({"pregunta": "¿Qué es LangChain?"})
    assert isinstance(result, str)
    assert len(result) > 0


def test_chain_returns_expected_response():
    expected = "Respuesta de prueba."
    chain = build_chain([expected])
    result = chain.invoke({"pregunta": "cualquier pregunta"})
    assert result == expected
