# Conceptual Similarity

Asserting for conceptual similarity allows developers to ensure that the expected answer and the generated answer are similar in terms of what is mentioned (even if the overall message can vary quite a bit.)

## What is it?

- Neural network embeddings are designed to represent the semantic meaning of words or concepts in a continuous vector space. These embeddings aim to capture the relationships and similarities between words or concepts based on their intrinsic properties.
- Techniques like word2vec, GloVe, and BERT embeddings are trained to learn the meaning and relationships between words or concepts from large text corpora. They excel at capturing the underlying semantics and conceptual associations between words.
- These embeddings are often used in various natural language processing (NLP) tasks like word similarity, text classification, and sentiment analysis, where understanding the meaning and similarity of words or concepts is crucial.

## Assert Conceptual Similarity

```python
from deepeval.test_utils import assert_conceptual_similarity

assert_conceptual_similarity(
    output="python is a programming language",
    expected_output="Python is a snake.",
    success_threshold=0.3
)
```

## Conceptual Similarity As A Metric

```python
from deepeval.metrics.conceptual_similarity import ConceptualSimilarityMetric

metric = ConceptualSimilarityMetric(success_threshold=0.3)
score = metric.measure(text_1="Python is a programming language.", text_2="Python is a snake.")
metric.is_successful()
```

### Parameters

- `success_threshold` - the minimum score required a metric to be successful
