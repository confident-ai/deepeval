# Answer Relevancy

For question-answering applications, we provide a simple interface for ensuring question-answering relevancy.

```python
from deepeval.test_utils import assert_answer_relevancy
query = "What is Python?"
answer = "Python is a programming language?"
assert_answer_relevancy(query, answer, success_threshold=0.5)
```

## Using answer relevancy as a metric

If you would instead like a score of how relevant an answer is to a query, simply call the metric class.

```python
from deepeval.metrics.answer_relevancy import AnswerRelevancy
scorer = AnswerRelevancy(success_threshold=0.5)
scorer.measure(query=query, answer=answer)
# Returns a floating point number between 0 and 1
```

### Parameters

- `success_threshold` refers to the minimum score for this to be considered relevant
