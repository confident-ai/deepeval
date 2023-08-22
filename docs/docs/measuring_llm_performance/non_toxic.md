# Toxicness

A large number of LLMs can become toxic after finetuning.

You can check if it is toxic if you

## Assert Non-Toxic

```python
from deepeval.metrics.toxic_classifier import assert_non_toxic

assert_non_toxic(text="Who is that?")
```

## Non-Toxicness as a Metric

```python
from deepeval.metrics.toxic_classifier import NonToxicMetric

metric = NonToxicMetric()
score = metric.measure(text=generated_text)


```

### How it is measured

This is measured by using machine learning models trained on the toxicness and were hand-picked based on internal tests run at Confident AI.
