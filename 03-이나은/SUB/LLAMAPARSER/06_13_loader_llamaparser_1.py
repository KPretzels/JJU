# -*- coding: utf-8 -*-
"""06_13_LOADER_LLAMAPARSER-1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_TN6JkXeFai2JT9zyhdLEV18gcPMfw34

## Upstage
"""

pip install -U langchain-upstage

import os
from langchain_upstage import UpstageLayoutAnalysisLoader

# 환경 변수에 API 키 설정
os.environ["UPSTAGE_API_KEY"] = ""

# 파일 경로
file_path = "/content/3-3_상관분석_수정.pdf"

# 문서 로더 설정
loader = UpstageLayoutAnalysisLoader(
    file_path=file_path,
    output_type="text",
    split="page",
    use_ocr=True,
    exclude=["header", "footer"]
)

# 문서 로드
docs = loader.load()

# 결과 출력
for doc in docs[:3]:
    print(doc)

"""## Llamaparser"""

# 설치
!pip install -qU llama-index-core llama-parse llama-index-readers-file python-dotenv langchain_community

import os
import nest_asyncio

LLAMA_CLOUD_API_KEY = ''

nest_asyncio.apply()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# 파서 설정
parser = LlamaParse(
    result_type="markdown",  # "markdown"과 "text" 사용 가능
    num_workers=8,  # worker 수 (기본값: 4)
    verbose=True,
    language="ko",
    api_key = LLAMA_CLOUD_API_KEY
)

# SimpleDirectoryReader를 사용하여 파일 파싱
file_extractor = {".pdf": parser}

# LlamaParse로 파일 파싱
documents = SimpleDirectoryReader(
    input_files=["/content/3-3_상관분석_수정.pdf"],
    file_extractor=file_extractor,
).load_data()

# 페이지 수 확인
len(documents)

# 랭체인 도큐먼트로 변환
docs = [doc.to_langchain_format() for doc in documents]

# metadata 출력
docs[0].metadata

OPENAI_API_KEY = ""

documents = LlamaParse(
    use_vendor_multimodal_model = True,
    vendor_multimodal_model_name = 'openai-gpt4o',
    vendor_multimoal_api_key = OPENAI_API_KEY,
    api_key = LLAMA_CLOUD_API_KEY,
    result_type = 'markdown',
    language = 'ko',
    skip_diagonal_text = True,
)



# parsing 된 결과
parsed_docs = documents.load_data(file_path="/content/3-3_상관분석_수정.pdf")

# langchain 도큐먼트로 변환
docs = [doc.to_langchain_format() for doc in parsed_docs]

parsing_instruction = (
    "You are parsing a brief of AI Report. Please extract tables in markdown format."
)

parser = LlamaParse(
    use_vendor_multimodal_model=True,  # 오타 수정: use_vendor_muitimoal_model -> use_vendor_multimodal_model
    vendor_multimodal_model_name='openai-gpt4o',
    vendor_multimodal_api_key='',
    result_type='markdown',
    language='ko',
    skip_diagonal_text=True,
    parsing_instruction=parsing_instruction,
    api_key=LLAMA_CLOUD_API_KEY  # API 키 추가
)

parser

# markdown 형식으로 추출된 테이블 확인
print(docs[-2].page_content)

