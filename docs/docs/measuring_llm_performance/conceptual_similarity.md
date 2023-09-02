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
from deepeval.metrics.conceptual_similarity import assert_conceptual_similarity

assert_conceptual_similarity(
    output="python is a programming language",
    expected_output="Python is a snake.",
    minimum_score=0.3
)
```

## Conceptual Similarity As A Metric

```python
from deepeval.metrics.conceptual_similarity import ConceptualSimilarityMetric

metric = ConceptualSimilarityMetric(minimum_score=0.3)
score = metric.measure(text_1="Python is a programming language.", text_2="Python is a snake.")
metric.is_successful()
```

### Parameters

- `minimum_score` - the minimum score required a metric to be successful
