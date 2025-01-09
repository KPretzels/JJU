from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')

from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt

llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout = None,
    max_retries=2,
    # api_key = OPENAI_API_KEY,  # os 설정을 했으면 없어도 됨.
    # base_url = '...',
)

prompt_yaml = load_prompt(r'C:\Users\kyh97\OneDrive\문서\GitHub\JJU\09_권윤형\WEEK3\PROMPTTEMPLATE.yaml', encoding='utf-8')

chain = prompt_yaml | llm
result = chain.invoke({'program': 'python'})

print(result.content)

output_path = r'C:\Users\kyh97\OneDrive\문서\GitHub\JJU\09_권윤형\WEEK3\ANSWERFROMLLM.txt'

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(result.content)