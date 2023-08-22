# Define Your Own Metric

By default, we support the following metrics:

- BertScoreMetric (simply set `metric="BertScoreMetric"`)
- Entailment Score (simply set `metric="entailment"`)
- Exact string match (simply set `metric="exact"`)

You can define a custom metric by defining the `measure` and `is_successful` functions and inheriting the base `Metric` class. An example is provided below.

```python
import asyncio
from deepeval.metrics.metric import Metric

class LengthMetric(Metric):
    """This metric checks if the output is more than 3 letters"""
    def __init__(self, minimum_length: int=3):
        self.minimum_length = minimum_length

    def __call__(self, text: str):
        # sends to server
        score = self.measure(text)
        # Optional: Logs it to the server
        asyncio.create_task(
            self._send_to_server(
                metric_score=score,
                metric_name=self.__name__,
                query=text,
                success = self.success
            )
        )
        return self.measure(text)

    def measure(self, text: str):
        self.success = len(x) > self.minimum_length
        return a > b

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Length"

metric = LengthMetric()
```
