from .metric import Metric
from deepeval.test_case import LLMTestCase


class LengthMetric(Metric):
    """This metric returns the length of the output if the output exists and removes any spaces on each end"""

    def __init__(self, minimum_length: int = 0):
        self.minimum_length = minimum_length

    def measure(self, test_case: LLMTestCase):
        # sends to server
        score = len(test_case.output.strip())
        self.success = score > self.minimum_length
        return score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "AnswerLength"
