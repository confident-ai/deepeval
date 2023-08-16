# Custom Metrics

By default, we support the following metrics: 

- BertScore (simply set `metric="bertscore"`)
- Entailment Score (simply set `metric="entailment"`)
- Exact string match (simply set `metric="exact"`)

You can define a custom metric by defining the `measure` and `is_successful` functions and inheriting the base `Metric` class. An example is provided below.

```python

from deepeval.metric import Metric

class CustomMetric(Metric):
    def measure(self, a, b):
        return a > b

    def is_successful(self):
        return True

metric = CustomMetric()

```
