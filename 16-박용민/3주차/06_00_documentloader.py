# -*- coding: utf-8 -*-
"""06-00 DocumentLoader.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10jl6KEmXpu9z8lGfC8LN_d_BiosFIxA6

# 사용 문서 링크
https://spri.kr/posts/view/23669
"""

from langchain_core.documents import Document

document= Document(page_content="안녕하세요? 이건 랭체인의 도큐먼트 입니다")

document.__dict__

document.metadata['source'] = '나'
document.metadata['page'] = 1
document.metadata['224444'] = '숫자'

FILE_PATH = '/content/data/SPRI_AI_Brief_2023년12월호_F.pdf'

pip install langchain_community

pip install pypdf

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(FILE_PATH)

docs = loader.load()

len(docs)

docs[0]

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)

split_docs = loader.load_and_split(text_splitter=text_splitter)

print(f'문서의 길이:{[len(split_docs)]}')

split_docs[20]

loader.lazy_load()

for doc in loader.lazy_load():
  print(doc)
  print('=' * 20)

adocs = loader.aload()

await adocs