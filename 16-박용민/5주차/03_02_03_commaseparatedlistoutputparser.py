# -*- coding: utf-8 -*-
"""03-02-03_CommaSeparatedListOutputParser.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NfivZ8noEaeFxtEU2qvzcqDsOvRDZzZZ
"""

import os

os.environ["OPENAI_API_KEY"] = "api-key"
os.environ["LANGCHAIN_API_KEY"] = "api-key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "http://api.smith.langchain.com"
os.environ["LANGCHAINPROJECT"] = "03-02-03"

!pip install langchain_openai

from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()
prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables={"subject"},
    partial_variables={"format_instructions": format_instructions},

)
model = ChatOpenAI(temperature=0)

prompt

chain = prompt | model | output_parser

response = chain.invoke({'subject': "한국인이 좋아하는 아이스크림 5종류"})

type(response)

