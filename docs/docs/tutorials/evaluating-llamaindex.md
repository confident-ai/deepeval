# Evaluating LlamaIndex

LlamaIndex connects data sources with queries and responses. It provides an opinionated framework for Retrieval-Augmented Generation.

## Installation and Setup

```sh
pip install -q -q llama-index
pip install -U deepeval
```

Once installed , you can get set up and start writing tests.

```sh
# Optional step: Login to get a nice dashboard for your tests later!
# During this step - make sure to save your project as llama
deepeval login
```

## Use With Your LlamaIndex

DeepEval integrates nicely with LlamaIndex's `ResponseEvaluator` class. Below is an example of the factual consistency documentation.

```python

from llama_index.response.schema import Response
from typing import List
from llama_index.schema import Document
from deepeval.metrics.factual_consistency import FactualConsistencyMetric

from llama_index import (
    TreeIndex,
    VectorStoreIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    Response,
)
from llama_index.llms import OpenAI
from llama_index.evaluation import ResponseEvaluator

import os
import openai

api_key = "sk-XXX"
openai.api_key = api_key

gpt4 = OpenAI(temperature=0, model="gpt-4", api_key=api_key)
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)
evaluator_gpt4 = ResponseEvaluator(service_context=service_context_gpt4)

```

#### Getting a lLamaHub Loader

```python
from llama_index import download_loader

WikipediaReader = download_loader("WikipediaReader")

loader = WikipediaReader()
documents = loader.load_data(pages=['Tokyo'])
tree_index = TreeIndex.from_documents(documents=documents)
vector_index = VectorStoreIndex.from_documents(
    documents, service_context=service_context_gpt4
)
```

We then build an evaluator based on the `BaseEvaluator` class that requires an `evaluate` method.

In this example, we show you how to write a factual consistency check.

```python
from deepeval.test_case import LLMTestCase
class FactualConsistencyResponseEvaluator:
  def get_context(self, response: Response) -> List[Document]:
    """Get context information from given Response object using source nodes.

    Args:
        response (Response): Response object from an index based on the query.

    Returns:
        List of Documents of source nodes information as context information.
    """
    context = []

    for context_info in response.source_nodes:
        context.append(Document(text=context_info.node.get_content()))

    return context

  def evaluate(self, response: Response) -> str:
    """Evaluate factual consistency metrics
    """
    answer = str(response)
    metric = FactualConsistencyMetric()
    context = self.get_context(response)
    context = " ".join([d.text for d in context])
    test_case = LLMTestCase(context=context, output=answer)
    score = metric.measure(test_case=test_case)
    if metric.is_successful():
        return "YES"
    else:
        return "NO"

evaluator = FactualConsistencyResponseEvaluator()
```

You can then evaluate as such:

```python
query_engine = tree_index.as_query_engine()
response = query_engine.query("How did Tokyo get its name?")
eval_result = evaluator.evaluate(response)
```
