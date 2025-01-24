from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')

doc_list = [
    'I like apples',
    'I like apple company',
    "I like apple's iphone",
    'Apple is my favorite company',
    "I like apple's ipand",
    "I like apple's macbook"
]

from langchain_core.runnables import ConfigurableField
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# bm25 retriever와 faiss retriever를 초기화합니다.
bm25_retriever = BM25Retriever.from_texts(
    doc_list,
)
bm25_retriever.k = 1  # BM25Retriever의 검색 결과 개수를 1로 설정합니다.

embedding = OpenAIEmbeddings()  # OpenAI 임베딩을 사용합니다.

faiss_vectorstore = FAISS.from_texts(
    doc_list,
    embedding,
)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 1})

ensemble_retriever = EnsembleRetriever(
    # 리트리버 목록을 설정합니다. 여기서는 bm25_retriever와 faiss_retriever를 사용합니다.
    retrievers=[bm25_retriever, faiss_retriever],
).configurable_fields(
    weights=ConfigurableField(
        # 검색 매개변수의 고유 식별자를 설정합니다.
        id="ensemble_weights",
        # 검색 매개변수의 이름을 설정합니다.
        name="Ensemble Weights",
        # 검색 매개변수에 대한 설명을 작성합니다.
        description="Ensemble Weights",
    )
)

config = {"configurable": {"ensemble_weights": [0.7, 0.3]}}

# config 매개변수를 사용하여 검색 설정을 지정합니다.
docs = ensemble_retriever.invoke("my favorite fruit is apple", config=config)
print(docs)  # 검색 결과인 docs를 출력합니다.