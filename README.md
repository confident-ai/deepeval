# LLMEvals

LLMEvals provides a Pythonic way to run offline evaluations on your LLM pipelines so you can launch comfortably into production.

You can get started with adding an a ground truth like so:

# Installation

```
pip install llmevals
```

# QuickStart

## Individual Test Cases

```python
# test_example.py
from evals.test_utils import assert_match, TestEvalCase, tags

def generate_llm_output(input: str):
    expected_output = "Our customer success phone line is 1200-231-231."
    return expected_output

class TestLLM(TestEvalCase):
    @tags(tags=["customer success"])
    def test_llm_output(self):
        input = "What is the customer success phone line?"
        expected_output = "Our customer success phone line is 1200-231-231."
        output = generate_llm_output(input)
        assert_match(output, expected_output, metric="exact")

Running tests ... ✅
```

Once you have set that up, you can simply call pytest

```bash
python -m pytest test_example.py
```

## Bulk Test Cases

You can run a number of test cases which you can define either through CSV
or through our hosted option.

```python
from evals import BulkTestRunner, TestCase

class BulkTester(BulkTestRunner):
    @property
    def bulk_test_cases(self):
        return [
            TestCase(
                input="What is the customer success number",
                expected_output="1800-213-123",
                metrics=["default"]
            )
        ]

tester = BulkTester()
tester.run()
```

## Setting up metrics

### Setting up custom metrics

TODO

## Setting up a LangChain pipeline

```python
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

loader = TextLoader("../../state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0
)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

from evals.metric import SimilarityMetric
pipeline = Pipeline(
    "langchain-example", result_fn=qa.run
)
evaluator.evaluate(
    pipeline=pipeline, 
    metric=metric
)
```

# Synthetic Query Generation 

Generating synthetic queries allows you to quickly evaluate the queries related to your prompts.
We help developers get up and running with a lot of example queries.

```python
# Loads the synthetic query model to generate them based on data you get.
# These automatically create synthetic queries and adds them as ground truth 
# for users
evaluator.generate_queries(
    texts=["Our customer success phone line is 1200-231-231"],
    tags=["customer success"]
)
```

# Dashboard

Once you have added a ground truth, you should be able to see a dashboard that contains information about the pipeline and the run.

![assets/app.png](assets/app.png)
