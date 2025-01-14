# -*- coding: utf-8 -*-
"""07_06_MarkdownHeaderTextSplitter.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vzfa5prMAhzYCjNkqIovm5EqoEc1XNMt

## 1.MarkdownHeaderTextSplitter
"""

from langchain_text_splitters import MarkdownHeaderTextSplitter

# 마크다운 형식의 문서를 문자열로 정의합니다.
markdown_document = '# Foo\n\n ## Bar\n\nHi this is Jim  \nHi this is Joe\n\n ## Baz\n\n Hi this is Molly'
print(markdown_document)

"""# Splitter 생성"""

headers_to_split_on = [  # 문서를 분할할 헤더 레벨과 해당 레벨의 이름을 정의합니다.
    (
        "#",
        "Header 1",
    ),  # 헤더 레벨 1은 '#'로 표시되며, 'Header 1'이라는 이름을 가집니다.
    (
        "##",
        "Header 2",
    ),  # 헤더 레벨 2는 '##'로 표시되며, 'Header 2'라는 이름을 가집니다.
    (
        "###",
        "Header 3",
    ),  # 헤더 레벨 3은 '###'로 표시되며, 'Header 3'이라는 이름을 가집니다.
]

# 마크다운 헤더를 기준으로 텍스트를 분할하는 MarkdownHeaderTextSplitter 객체를 생성합니다.
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = headers_to_split_on)

# markdown_document를 헤더를 기준으로 분할하여 md_header_splits에 저장합니다.
md_header_splits = markdown_splitter.split_text(markdown_document)

# 분할된 결과를 출력합니다.
for header in md_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")

markdown_splitter = MarkdownHeaderTextSplitter(
    # 분할할 헤더를 지정합니다.
    headers_to_split_on=headers_to_split_on,
    # 헤더를 제거하지 않도록 설정합니다.
    strip_headers=False,
)
# 마크다운 문서를 헤더를 기준으로 분할합니다.
md_header_splits = markdown_splitter.split_text(markdown_document)
# 분할된 결과를 출력합니다.
for header in md_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")

"""# 자소서 markdown"""

# Open and read the contents of the file
with open('/content/testresume.md', 'r') as f:
    file = f.read()

# Display the content
print(file)

# 문서를 분할할 헤더 레벨과 해당 레벨의 이름을 정의합니다.
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# MarkdownHeaderTextSplitter 객체 생성 (헤더 기준으로 분할)
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = headers_to_split_on)

# 마크다운 문서를 헤더 기준으로 분할
md_header_splits = markdown_splitter.split_text(file)

# 분할된 결과 출력
for header in md_header_splits:
    print(f"{header.page_content}")   # 분할된 각 섹션의 내용
    print(f"{header.metadata}", end="\n=====================\n")  # 해당 섹션의 메타데이터

markdown_splitter = MarkdownHeaderTextSplitter(
    # 분할할 헤더를 지정합니다.
    headers_to_split_on=headers_to_split_on,
    # 헤더를 제거하지 않도록 설정합니다.
    strip_headers=False,
)
# 마크다운 문서를 헤더를 기준으로 분할합니다.
md_header_splits = markdown_splitter.split_text(file)
# 분할된 결과를 출력합니다.
for header in md_header_splits:
    print(f"{header.page_content}")
    print(f"{header.metadata}", end="\n=====================\n")

