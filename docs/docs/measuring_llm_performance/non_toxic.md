# Toxicness

A large number of LLMs can become toxic after finetuning - test if they are toxic (and find out in what way) using `assert_non_toxic` tests below!

## Assert Non-Toxic

```bash
pip install deepeval[toxic]
```

```python
from deepeval.metrics.toxic_classifier import assert_non_toxic

assert_non_toxic(text="Who is that?")
```

## Non-Toxicness as a Metric

```python
from deepeval.metrics.toxic_classifier import NonToxicMetric
from deepeval.run_test import run_test, assert_test
from deepeval.test_case import LLMTestCase

metric = NonToxicMetric()
test_case = LLMTestCase(input="This is an example query", output=output)

# If you want to run a test, log it and check results
run_test(test_case, metrics=[metric])

# If you want to make sure a test passes
assert_test(test_case, metrics=[metric])
```

### How it is measured

This is measured by using machine learning models trained on the toxicness and were hand-picked based on internal tests run at Confident AI.

Under the hood, DeepEval uses `detoxify` package to measure toxicness.
