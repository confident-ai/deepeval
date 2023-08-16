# Evaluating LangChain

Most developers are currently building their retrieval pipelines using popular frameworks such as LangChain. This tutorial shows how you can evaluate LangChain pipelines using DeepEval.

## Defining a QA Chain

First, let us define a Retrieval QA pipeline.

```bash
pip install openai chromadb langchain tiktoken
```

```python
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

text_file_url = "https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt"

openai_api_key = "sk-XXX"

with open("state_of_the_union.txt", "w") as f:
  response = requests.get(text_file_url)
  f.write(response.text)

loader = TextLoader("state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key="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")
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(
  llm=OpenAI(openai_api_key=openai_api_key), chain_type="stuff"
  retriever=docsearch.as_retriever()
)

# Providing a new question-answering pipeline
result = qa.run(query)

```




