# Evaluating LangChain

Most developers are currently building their retrieval pipelines using popular frameworks such as LangChain. This tutorial shows how you can evaluate LangChain pipelines using DeepEval.

## Defining a QA Chain

First, let us define a Retrieval QA pipeline.

```bash
pip install openai chromadb langchain
```

```python
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

text_file_url = "https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt"

with open("state_of_the_union.txt", "w") as f:
  response = requests.get(text_file_url)
  f.write(response.text)
```
