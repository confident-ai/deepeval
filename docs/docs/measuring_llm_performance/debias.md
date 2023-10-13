# Bias

LLMs can become highly biased after finetuning from from any RLHF or optimizations.

Bias, however, is a very vague term so the paper focuses on bias in the following areas (as shown in the example).

- Gender (e.g. "All man hours in his area of responsibility must be approved.")
- Age (e.g. "Apply if you are a recent graduate.")
- Racial/Ethnicity (e.g. "Police are looking for any black males who may be involved in this case.")
- Disability (e.g. "Genuine concern for the elderly and handicapped")
- Mental Health (e.g. "Any experience working with retarded people is required for this job.")
- Religion
- Education
- Political ideology

## Assert UnBiased

```python
from deepeval.metrics.bias_classfier import is_unbiased

assert is_unbiased(text="I can presume bias only exists in Tanzania")
```

## UnBiased as a Metric

```python
from deepeval.metrics.bias_classifier import UnBiasedMetric
from deepeval.test_case import LLMTestCase
from deepeval import run_test, assert_test

metric = UnBiasedMetric()
test_case = LLMTestCase(
    input="This is an example input",
    actual_output="Devil wing is evil."
)
run_test(test_case, [metric])
# Prints out score for bias measure, 1 being highly biased 0 being unbiased

```

### How it is measured

This is measured according to tests with logic following this paper https://arxiv.org/pdf/2208.05777.pdf

DeepEval uses DBias under the hood to measure bias.
