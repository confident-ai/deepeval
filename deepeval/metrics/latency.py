from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class LatencyMetric(BaseMetric):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase):
        self.success = test_case.latency <= self.threshold
        self.score = test_case.latency
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Latency"
