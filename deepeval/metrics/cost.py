from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class CostMetric(BaseMetric):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        self.success = test_case.cost <= self.threshold
        self.score = test_case.cost
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Cost"
