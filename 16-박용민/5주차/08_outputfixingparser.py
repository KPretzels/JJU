# -*- coding: utf-8 -*-
"""08-OutputFixingParser.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/106hi-GbbAbBUKk9uXrTV62FFWZW0I8RU
"""

import os

os.environ["OPENAI_API_KEY"] = "api-key"
os.environ["LANGCHAIN_API_KEY"] = "api-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "http://api.smith.langchain.com"
os.environ["LANGCHAINPROJECT"] = "03-08"



from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Pydantic 모델 정의
class Actor(BaseModel):
    name: str = Field(description="name of the actor")
    film_names: List[str] = Field(description="list of names of films they starred in")

actor_query = "Genarate the filmography for a random actor."
# PydanticOutputParser 초기화
parser = PydanticOutputParser(pydantic_object=Actor)

base_Actor="{'name' : 'Tom Hanks', 'film_names': ['Forest Gump']}"

parser.parse(base_Actor)

from langchain.output_parsers import OutputFixingParser

new_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI(model="gpt-4o"))

final_Actor=new_parser.parse(base_Actor)

final_Actor

