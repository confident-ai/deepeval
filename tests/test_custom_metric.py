"""Test for custom metrics in Python
"""

import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics.metric import Metric


class LengthMetric(Metric):
    """This metric checks if the output is more than 3 letters"""

    def __init__(self, minimum_length: int = 3):
        self.minimum_length = minimum_length

    def measure(self, test_case: LLMTestCase):
        # sends to server
        text = test_case.output
        score = len(text)
        self.success = score > self.minimum_length
        # Optional: Logs it to the server
        self.log(
            query=text,
            score=score
            / 100,  # just to have something here - should be between 0 and 1
            success=self.success,
        )
        return score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Length"


def test_length_metric():
    metric = LengthMetric()
    test_case = LLMTestCase(
        output="This is a long sentence that is more than 3 letters"
    )
    metric.measure(test_case)
    assert metric.is_successful()
