UPSTAGE_API_KEY = ''

import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "test_JJU_YuSangMin"

from langchain_upstage import UpstageLayoutAnalysisLoader

file_path = '/content/data/SPRI_AI_Brief_2023년12월호_F.pdf'

loader = UpstageLayoutAnalysisLoader(
    file_path,
    output_type = 'html',
    split = 'page',
    use_ocr = True,
    exclude = ["header", 'footer'],
    api_key= UPSTAGE_API_KEY

)


docs1 = loader.load()

for doc in docs1[6]:
  print(doc)

loader = UpstageLayoutAnalysisLoader(
    file_path,
    output_type = 'html',
    split = 'element',
    use_ocr = True,
    exclude = ["header", 'footer'],
    api_key= UPSTAGE_API_KEY

)


docs2 = loader.load()

for doc in docs2[6]:
  print(doc)