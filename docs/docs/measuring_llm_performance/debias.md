# Bias

LLMs can become highly biased after finetuning from from any RLHF or optimizations otherwise

## Assert UnBiased

```python
from deepeval.metrics.bias_classfier import assert_unbiased

assert_unbiased(text="I can presume bias only exists in Tanzania")
```

## UnBiased as a Metric

```python
from deepeval.metrics.bias_classifier import UnBiasedMetric

metric = UnBiasedMetric()
score = metric.measure(text=generated_text)
score
# Prints out score for bias measure, 1 being highly biased 0 being unbiased

```

### How it is measured

This is measured according to tests with logic following this paper https://arxiv.org/pdf/2208.05777.pdf

DeepEval uses DBias in this case
