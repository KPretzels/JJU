from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')

# from langchain_core.prompts import load_prompt

# yaml_file = load_prompt(r'C:\Users\kyh97\OneDrive\문서\GitHub\JJU\09_권윤형\SUB\FewShot\data.yaml', encoding='utf-8')
# print(yaml_file)

from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate

examples = [
    {
        "question": "스티브 잡스와 아인슈타인 중 누가 더 오래 살았나요?",
        "answer": """
                  이 질문에 추가 질문이 필요한가요: 예.
                  추가 질문: 스티브 잡스는 몇 살에 사망했나요?
                  중간 답변: 스티브 잡스는 56세에 사망했습니다.
                  추가 질문: 아인슈타인은 몇 살에 사망했나요?
                  중간 답변: 아인슈타인은 76세에 사망했습니다.
                  최종 답변은: 아인슈타인
                  """,
    },
    {
        "question": "네이버의 창립자는 언제 태어났나요?",
        "answer": """
                  이 질문에 추가 질문이 필요한가요: 예.
                  추가 질문: 네이버의 창립자는 누구인가요?
                  중간 답변: 네이버는 이해진에 의해 창립되었습니다.
                  추가 질문: 이해진은 언제 태어났나요?
                  중간 답변: 이해진은 1967년 6월 22일에 태어났습니다.
                  최종 답변은: 1967년 6월 22일
                  """,
    },
    {
        "question": "율곡 이이의 어머니가 태어난 해의 통치하던 왕은 누구인가요?",
        "answer": """
                  이 질문에 추가 질문이 필요한가요: 예.
                  추가 질문: 율곡 이이의 어머니는 누구인가요?
                  중간 답변: 율곡 이이의 어머니는 신사임당입니다.
                  추가 질문: 신사임당은 언제 태어났나요?
                  중간 답변: 신사임당은 1504년에 태어났습니다.
                  추가 질문: 1504년에 조선을 통치한 왕은 누구인가요?
                  중간 답변: 1504년에 조선을 통치한 왕은 연산군입니다.
                  최종 답변은: 연산군
                  """,
    },
    {
        "question": "올드보이와 기생충의 감독이 같은 나라 출신인가요?",
        "answer": """
                  이 질문에 추가 질문이 필요한가요: 예.
                  추가 질문: 올드보이의 감독은 누구인가요?
                  중간 답변: 올드보이의 감독은 박찬욱입니다.
                  추가 질문: 박찬욱은 어느 나라 출신인가요?
                  중간 답변: 박찬욱은 대한민국 출신입니다.
                  추가 질문: 기생충의 감독은 누구인가요?
                  중간 답변: 기생충의 감독은 봉준호입니다.
                  추가 질문: 봉준호는 어느 나라 출신인가요?
                  중간 답변: 봉준호는 대한민국 출신입니다.
                  최종 답변은: 예
                  """,
    },
]

example_prompt = PromptTemplate.from_template(
    "Question:\n{question}\nAnswer:\n{answer}"
)

prompt = FewShotPromptTemplate(
    examples = examples,
    example_prompt=example_prompt,
    suffix="Question:\n{question}\nAnswer",
    input_variables=["question"],
)

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    temperature=0,   # 창의성
    model = 'gpt-4o' # 모델명
    )

chain = prompt | llm

print(chain.invoke(input={'question':'전주대학교 인공지능학과가 설립됐을 때의 대통령이 속해있던 정당 이름은?'}).content)