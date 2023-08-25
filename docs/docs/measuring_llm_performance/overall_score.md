# Overall Score

Alert score checks if a generated output is good or bad. It automatically checks for:

- Factual consistency
- Answer relevancy

It then takes the mean of these scores.

## Assert Alert Score

```python
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

```python
from deepeval.metrics.overall_score import OverallScoreMetric
metric = OverallScoreMetric()
score = metric.measure(
    query="Who won the FIFA World Cup in 2018?",
    output="French national football team",
    expected_output="The FIFA World Cup in 2018 was won by the French national football team.",
    context="The FIFA World Cup in 2018 was won by the French national football team. They defeated Croatia 4-2 in the final match to claim the championship.",
    minimum_score=0.3
)
score
```

### How it is measured

It takes the mean of factual consistency, answer relevancy and toxicness to give an overall average score.
