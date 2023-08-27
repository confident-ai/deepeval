# Define Your Own Metric

You can define a custom metric by defining the `measure` and `is_successful` functions and inheriting the base `Metric` class. An example is provided below.

:::note

As of right now, we do not currently support custom metrics for our dashboard but this will be supported in an upcoming version - we apologise for the wait! For any requests of additional metrics, please feel free to e-mail jacky@confident-ai.com

:::

```python
from deepeval.metrics.metric import Metric

class LengthMetric(Metric):
    """This metric checks if the output is more than 3 letters"""
    def __init__(self, minimum_length: int=3):
        self.minimum_length = minimum_length

    def measure(self, text: str):
        # sends to server
        score = len(text)
        self.success = score > self.minimum_length
        # Optional: Logs it to the server for viewing
        self.log(
            query=text,
            score=score/100, # just to have something here - should be between 0 and 1
            success=self.success
        )
        return score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Length"

metric = LengthMetric()

```
