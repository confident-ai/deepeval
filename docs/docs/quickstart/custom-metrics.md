# Define Your Own Metric

You can define a custom metric by defining the `measure` and `is_successful` functions and inheriting the base `Metric` class. An example is provided below.

```python
from deepeval.metrics.metric import Metric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import run_test

class LengthMetric(Metric):
    """This metric checks if the output is more than 3 letters"""
    def __init__(self, minimum_length: int=3):
        self.minimum_length = minimum_length

    def measure(self, test_case: LLMTestCase):
        # sends to server
        score = len(test_case.output)
        self.success = score > self.minimum_length
        return score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Length"

metric = LengthMetric()

# Defining a custom test case
test_case = LLMTestCase(input="This is an example input", output="This is an example output")
run_test(test_case, metrics=[metric])
```
