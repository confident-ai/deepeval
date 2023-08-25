# Toxicness

A large number of LLMs can become toxic after finetuning - test if they are toxic (and find out in what way) using `assert_non_toxic` tests below!

## Assert Non-Toxic

```python
from deepeval.metrics.toxic_classifier import assert_non_toxic

assert_non_toxic(text="Who is that?")
```

## Non-Toxicness as a Metric

```python
from deepeval.metrics.toxic_classifier import NonToxicMetric

metric = NonToxicMetric()
score = metric.measure(text=output)
score
# Prints out a dictionary of values showing the scores for each trait

```

### How it is measured

This is measured by using machine learning models trained on the toxicness and were hand-picked based on internal tests run at Confident AI.

Under the hood, DeepEval uses `detoxify` package to measure toxicness.
