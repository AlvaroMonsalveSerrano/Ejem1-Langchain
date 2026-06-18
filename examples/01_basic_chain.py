"""
Ejemplo 01: Cadena básica con prompt + modelo + parser de salida.
Patrón fundamental de LangChain: PromptTemplate | LLM | OutputParser
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil que responde en español."),
        ("human", "{pregunta}"),
    ])

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    chain = prompt | model | parser

    respuesta = chain.invoke({"pregunta": "¿Qué es LangChain en una oración?"})
    print(respuesta)


if __name__ == "__main__":
    main()
