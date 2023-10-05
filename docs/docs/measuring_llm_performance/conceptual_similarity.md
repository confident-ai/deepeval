# Conceptual Similarity

Asserting for conceptual similarity allows developers to ensure that the expected answer and the generated answer are similar in terms of what is mentioned (even if the overall message can vary quite a bit.)

## How This Differs From Answer Relevancy

For example - when asked:

How big is this apple?
- 12 square feet
- The size of an orange

While they are both relevant and may not be hallucinating - it's important to note that they are conceptually different ways of answering.

## Assert Conceptual Similarity

```python
from deepeval.metrics.conceptual_similarity import is_conceptually_similar, assert_conceptually_similar

# If want the boolean value
assert is_conceptually_similar(
    output="python is a programming language",
    expected_output="Python is a snake.",
    minimum_score=0.3
)

# If you want native print statements
assert_conceptually_similar(
    output="python is a programming language",
    expected_output="Python is a snake.",
    minimum_score=0.3
)
```

## Conceptual Similarity As A Metric

```python
from deepeval.metrics.conceptual_similarity import ConceptualSimilarityMetric
from deepeval.run_test import run_test, assert_test
from deepeval.test_case import LLMTestCase

metric = ConceptualSimilarityMetric(minimum_score=0.3)
test_case = LLMTestCase(output=output, context=context)

# If you want to run a test, log it and check results
run_test(test_case, metrics=[metric])

# If you want to make sure a test passes
assert_test(test_case, metrics=[metric])
```

### Parameters

- `minimum_score` - the minimum score required a metric to be successful
