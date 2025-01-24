from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')

from langchain_core.prompts import load_prompt

prompt = load_prompt(r'C:\Users\kyh97\OneDrive\문서\GitHub\JJU\09_권윤형\WEEK4\0_CHATPROMPTTEMPLATE\PROMPT-CHATPROMPTTEMPLATE.yaml', encoding='utf-8')




#prompt = yaml_file.format(nation='미국')

llm = ChatOpenAI(
    temperature=0,   # 창의성
    model = 'gpt-4o' # 모델명
    )
chain = prompt | llm

print(chain.invoke({'nation':'로스앤젤레스'}).content)
#chain = prompt | llm

#print(chain.invoke(input={'question':'전주대학교 인공지능학과가 설립됐을 때의 대통령이 속해있던 정당 이름은?'}).content)