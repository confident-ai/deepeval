# Evaluating LangChain Retrieval QA

Most developers are currently building their retrieval pipelines using popular frameworks such as LangChain. This tutorial shows how you can evaluate LangChain pipelines using DeepEval.

## Defining a QA Chain

First, let us define a Retrieval QA pipeline.

```bash
pip install openai chromadb langchain tiktoken
```

```python
import requests
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

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(
  llm=OpenAI(openai_api_key=openai_api_key), chain_type="stuff",
  retriever=docsearch.as_retriever()
)

# Providing a new question-answering pipeline
query = "Who is the president?
result = qa.run(query)

```

## Creating an Evaluation Dataset

Now that we have defined the LangChain QA pipeline, let us start getting an evaluation dataset.

### Creating A Synthetic Dataset

In most cases, users won't have an evaluation dataset and need a synthetic dataset to start. You can achieve this easily by a Python function for this.

:::note

This model is quite big and may consume a lot of memory. In order to run this code quickly and not consume a lot of memory - we recommend using a solution like Modal to run this in the cloud.

:::

```python
from deepeval.dataset import create_evaluation_dataset_from_raw_text

ds = create_evaluation_dataset_from_raw_text(response.text)
ds.run_evaluation(completion_fn=qa.run)
```

This will output a text file with the following contents:

```bash
Test Passed    Metric Name                 Score  Output                                                                                                                                                                               Expected output                                                     Message
-------------  ---------------------  ----------  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  ------------------------------------------------------------------  -------------------------------------------------------------------
False          EntailmentScoreMetric  0.00545002  No, synthetic queries are not useful for scenarios where there is no data. Synthetic queries are used to generate test data to evaluate the performance of a system or application.  Synthetic queries are useful for scenraios where there is no data.  EntailmentScoreMetric was unsuccessful for
                                                                                                                                                                                                                                                                                                           Synthetic queries are useful for scenraios where there is no data.
                                                                                                                                                                                                                                                                                                           which should have matched
                                                                                                                                                                                                                                                                                                           Synthetic queries are useful for scenraios where there is no data.
```
