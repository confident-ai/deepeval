"""Test for custom metrics in Python
"""

from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.evaluator import assert_test


class LengthMetric(BaseMetric):
    """This metric checks if the output is more than 3 letters"""

    def __init__(self, minimum_score: int = 3):
        self.minimum_score = minimum_score

    def measure(self, test_case: LLMTestCase):
        # sends to server
        text = test_case.actual_output
        score = len(text)
        self.success = score > self.minimum_score
        # Optional: Logs it to the server
        return score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Length"


def test_length_metric():
    metric = LengthMetric()
    test_case = LLMTestCase(
        input="placeholder",
        actual_output="This is a long sentence that is more than 3 letters",
    )
    assert_test(test_case, [metric])
