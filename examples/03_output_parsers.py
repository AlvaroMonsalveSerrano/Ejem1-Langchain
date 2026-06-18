"""
Ejemplo 03: Output parsers — StrOutputParser, JsonOutputParser, PydanticOutputParser.
"""

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def ejemplo_str():
    chain = ChatPromptTemplate.from_template("Explica {tema} en una oración.") | model | StrOutputParser()
    print(chain.invoke({"tema": "los agentes de IA"}))


def ejemplo_json():
    prompt = ChatPromptTemplate.from_template(
        "Devuelve un JSON con los campos 'nombre', 'capital' y 'poblacion' para el país: {pais}. "
        "Responde SOLO con el JSON, sin texto adicional."
    )
    chain = prompt | model | JsonOutputParser()
    print(chain.invoke({"pais": "Colombia"}))


def ejemplo_pydantic():
    class Pelicula(BaseModel):
        titulo: str = Field(description="Título de la película")
        año: int = Field(description="Año de estreno")
        genero: str = Field(description="Género principal")

    parser = PydanticOutputParser(pydantic_object=Pelicula)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extrae la información de la película.\n{format_instructions}"),
        ("human", "{texto}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser
    resultado = chain.invoke({"texto": "Vi Inception de Christopher Nolan, estrenada en 2010, es ciencia ficción."})
    print(resultado)
    print(f"Título: {resultado.titulo}, Año: {resultado.año}")


if __name__ == "__main__":
    print("--- StrOutputParser ---")
    ejemplo_str()
    print("\n--- JsonOutputParser ---")
    ejemplo_json()
    print("\n--- PydanticOutputParser ---")
    ejemplo_pydantic()
