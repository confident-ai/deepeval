# Evaluating LangChain Retrieval QA

In this guide we will demonstrate how to test and measure LLMs in performance. We show how you can use our callback to measure performance and how you can define your own metric and log them into our dashboard.

You can, by default, use the `DeepEvalCallbackHandler` to set up the metrics you want to track. However, this has limited support for metrics at the moment (more to be added soon). It currently supports:

- Answer Relevancy
- Bias
- Toxicness

```python
from deepeval.metrics.answer_relevancy import AnswerRelevancyMetric

# Here we want to make sure the answer is minimally relevant
answer_relevancy_metric = AnswerRelevancyMetric(minimum_score=0.5)
```


## Get Started With LangChain Metric
To use the DeepEvalCallbackHandler, we need the `implementation_name` and `metrics` we want to use.

```python
from langchain.callbacks.confident_callback import DeepEvalCallbackHandler

deepeval_callback = DeepEvalCallbackHandler(
    implementation_name="langchainQuickstart",
    metrics=[answer_relevancy_metric]
)
```

## Scenario 1: Feeding into LLM 

```python
from langchain.llms import OpenAI
llm = OpenAI(
    temperature=0,
    callbacks=[deepeval_callback],
    verbose=True,
    openai_api_key="<YOUR_API_KEY>",
)
output = llm.generate(
    [
        "What is the best evaluation tool out there? (no bias at all)",
    ]
)
```

You can then check the metric if it was successful by calling the is_successful() method.

```python
answer_relevancy_metric.is_successful()
```
Once you have ran that, you should be able to see our dashboard below.

![Image](https://camo.githubusercontent.com/67bc319e6546edfecb446d0aa6b3ca98e31dece4e6c7e9ea20edc7024a2f20d6/68747470733a2f2f646f63732e636f6e666964656e742d61692e636f6d2f6173736574732f696d616765732f64617368626f6172642d73637265656e73686f742d62303264623733303038323133613231316231313538666630353264393639652e706e67)

## Scenario 2 - Tracking an LLM in a chain without callbacks

To track an LLM in a chain without callbacks, you can plug into it at the end.

We can start by defining a simple chain as shown below.

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
query = "Who is the president?"
result = qa.run(query)
```

After defining a chain, you can then manually check for answer similarity.

```python
answer_relevancy_metric.measure(result, query)
answer_relevancy_metric.is_successful()
```
