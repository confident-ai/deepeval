# Alert Score

Alert score checks if a generated output is good or bad. It automatically checks for:

- Factual consistency
- Answer relevancy
- Toxicness (coming soon)

It then takes the minimum of those scores to alert.

## Assert Alert Score

```python
from deepeval.metrics.alert_score import assert_alert_score
assert_alert_score(
    output="Who won the FIFA World Cup in 2018?",
    expected_output="French national football team",
    context="The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship.",
    minimum_score=0.3
)
```

## Metric

```python
from deepeval.metrics.alert_score import AlertScoreMetric
from deepeval.run_test import run_test, assert_test
from deepeval.test_case import LLMTestCase

metric = AlertScoreMetric()
test_case = LLMTestCase(input="This is an example input", output=output, context=context)

# If you want to run a test, log it and check results
run_test(test_case, metrics=[metric])

# If you want to make sure a test passes
assert_test(test_case, metrics=[metric])

```

### How it is measured

It takes the minimum of factual consistency, answer relevancy and toxicness so that the lowest score represents how bad it is.
