"""
Ejemplo 02: Variantes de PromptTemplate — system/human, few-shot, variables múltiples.
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


def ejemplo_basico():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Traduce el texto al {idioma}."),
        ("human", "{texto}"),
    ])
    chain = prompt | model | parser
    print(chain.invoke({"idioma": "francés", "texto": "Hola, ¿cómo estás?"}))


def ejemplo_few_shot():
    ejemplos = [
        {"input": "feliz", "output": "triste"},
        {"input": "alto", "output": "bajo"},
    ]
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=ejemplos,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Da el antónimo de la palabra."),
        few_shot_prompt,
        ("human", "{word}"),
    ])
    chain = prompt | model | parser
    print(chain.invoke({"word": "rápido"}))


if __name__ == "__main__":
    print("--- Básico ---")
    ejemplo_basico()
    print("--- Few-shot ---")
    ejemplo_few_shot()
