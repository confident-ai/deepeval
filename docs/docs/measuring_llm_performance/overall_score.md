# Overall Score

Overall score checks if a generated output is good or bad. It automatically checks for:

- Factual consistency (Test for factual consistency)
- Answer relevancy (Ensure the answer is relevant)
- Conceptual similarity (Know when queries are conceptually similar)

It then takes the mean of these scores.

## Assert Overall Score Is Above Threshold

For the overall score, the only required argument is `output` and the overall score will be calculated on as many of the above metrics as possible based on what is provided.

```python
# You can test for overall score based on the following:
from deepeval.metrics.overall_score import assert_overall_score
assert_overall_score(
    query="Who won the FIFA World Cup in 2018?",
    output="French national football team",
    expected_output="The FIFA World Cup in 2018 was won by the French national football team.",
    context="The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship.",
    minimum_score=0.3
)
```

## Metric

You can also dive into the lower level API which explores how to use this as a metric.

```python
from deepeval.metrics.overall_score import OverallScoreMetric
from deepeval.run_test import run_test, assert_test
from deepeval.test_case import LLMTestCase

metric = OverallScoreMetric(minimum_score=0.3)
test_case = LLMTestCase(
    input="Who won the FIFA World Cup in 2018?",
    actual_output="French national football team",
    expected_output="The FIFA World Cup in 2018 was won by the French national football team.",
    context="The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship.",
)

# If you want to run a test, log it and check results
run_test(test_case, metrics=[metric])

# If you want to make sure a test passes
assert_test(test_case, metrics=[metric])

```

### How it is measured

It takes the mean of factual consistency, answer relevancy and toxicness to give an overall average score.
