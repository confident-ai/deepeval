"""Test for custom metrics in Python
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval import assert_test


class LatencyMetric(BaseMetric):
    # This metric by default checks if the latency is greater than 10 seconds
    def __init__(self, max_seconds: float = 10):
        self.threshold = max_seconds

    def measure(self, test_case: LLMTestCase):
        # Set self.success and self.score in the "measure" method
        self.success = True
        if self.success:
            self.score = 1
        else:
            self.score = 0

        # You can also set a reason for the score returned.
        # This is particularly useful for a score computed using LLMs
        self.reason = None
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Latency"


def test_length_metric():
    metric = LatencyMetric()
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="This is a long sentence that is more than 3 letters",
    )
    # a_measure not implemented
    assert_test(test_case, [metric], run_async=False)
