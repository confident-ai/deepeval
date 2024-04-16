from typing import Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.telemetry import capture_metric_type


class LatencyMetric(BaseMetric):
    def __init__(self, max_latency: float):
        self.threshold = max_latency

    def measure(self, test_case: LLMTestCase):
        self.success = test_case.latency <= self.threshold
        self.score = test_case.latency
        capture_metric_type(self.__name__)
        return self.score

    async def a_measure(self, test_case: LLMTestCase):
        self.success = test_case.latency <= self.threshold
        self.score = test_case.latency
        capture_metric_type(self.__name__)
        return self.score

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Latency"
