from typing import Any, List
from .metrics.metric import Metric
from .metrics.randomscore import RandomMetric


class TestCase:
    def __init__(self, input: Any, expected_output: Any, metrics: List[Metric] = None):
        if metrics is None:
            self.metrics = [RandomMetric()]
        self.input = input
        self.expected_output = expected_output
