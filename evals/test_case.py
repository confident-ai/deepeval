import hashlib
from typing import Any, List
from .metrics.metric import Metric
from .metrics.randomscore import RandomMetric
from .metrics.entailment_metric import EntailmentScore


class TestCase:
    def __init__(
        self,
        input: Any,
        expected_output: Any,
        metrics: List[Metric] = None,
        id: str = None,
    ):
        if metrics is None:
            self.metrics = [EntailmentScore()]
        else:
            self.metrics = metrics
        self.input = input
        self.expected_output = expected_output
        if id is None:
            id_string = str(self.input) + str(self.expected_output)
            self.id = hashlib.md5(id_string.encode()).hexdigest()
        else:
            self.id = id
