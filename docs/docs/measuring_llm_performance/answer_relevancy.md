# Answer Relevancy

We provide a simple interface for ensuring question-answering relevancy powered by the cosine similarity between bi-encoder QA models. This is important when you have ground truths and LLM outputs and you want to make sure that the answers are relevant to the question.

## Assert Answer Relevancy

```python
from deepeval.metrics.answer_relevancy import assert_answer_relevancy
query = "What is Python?"
answer = "Python is a programming language?"
assert_answer_relevancy(query, output=answer, minimum_score=0.5)
```

### Parameters

- `minimum_score` refers to the minimum score for this to be considered relevant

## Answer Relevancy As A Metric

If you would instead like a score of how relevant an answer is to a query, simply call the metric class.

```python
from deepeval.metrics.answer_relevancy import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import run_test, assert_test
metric = AnswerRelevancyMetric(minimum_score=0.5)
test_case = LLMTestCase(query=query, output=answer)

# If you want to run a test, log it and check results
run_test(test_case, metrics=[metric])

# If you want to make sure a test passes
assert_test(test_case, metrics=[metric])
```

### Parameters

- `minimum_score` refers to the minimum score for this to be considered relevant

## How It is Measured

Answer relevancy is measured using DL models that are trained off MS-Marco dataset (which is a search engine dataset). The method to measure relevancy is that it encodes a query and an answer and then measures the cosine similarity. The vector space has been trained off query-answer MSMarco datasets to ensure high similarity between query and answer.
