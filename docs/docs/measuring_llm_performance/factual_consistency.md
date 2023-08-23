# Factual Consistency

Factual consistency refers to the accuracy and reliability of information presented in a piece of text, conversation, or any form of communication. It means that the information being conveyed is true, accurate, and aligns with established facts or reality. Factual consistency is crucial because it ensures that the information being shared is reliable and trustworthy. Inaccurate or inconsistent information can lead to misunderstandings, misinformation, and loss of credibility.

## Assert Factual Consistency

DeepEval offers an opinionated method for factual consistency based on entailment score.

```python
from deepeval.metrics.factual_consistency import assert_factual_consistency
assert_factual_consistency(
    output="Sarah spent the evening at the library, engrossed in a book.",
    context="After a long day at work, Sarah decided to go for a walk in the park to unwind. She put on her sneakers and grabbed her headphones before heading out. As she strolled along the path, she noticed families having picnics, children playing on the playground, and ducks swimming in the pond."
)
```

### Parameters

Diving into the arguments for `assert_factual_consistency`:

- `output` - the LLM generated text
- `context` - the text from which the LLM is supposed to reason and derive conclusions from

## Factual Consistency As A Metric

If you would instead like a score of how factually consistent the output is relative to the context.

```python
from deepeval.metrics.factual_consistency import FactualConsistencyMetric
metric = FactualConsistencyMetric(minimum_score=0.5)
metric.measure(output=output, context=context)
# Returns a floating point number between 0 and 1
```

### How It Is Measured

Factual consistency is measured using natural language inference models based on the output score of the entailment class that compare the ground truth and the context from which the ground truth is done.
