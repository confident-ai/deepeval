import hashlib
from typing import Any, List
from collections import UserList
from .metrics.metric import Metric
from .metrics.randomscore import RandomMetric
from .metrics.entailment_metric import EntailmentScoreMetric


class TestCase:
    def __init__(
        self,
        input: Any,
        expected_output: Any,
        metrics: List[Metric] = None,
        id: str = None,
    ):
        if metrics is None:
            self.metrics = [EntailmentScoreMetric(minimum_score=0.3)]
        else:
            self.metrics = metrics
        self.input = input
        self.expected_output = expected_output
        if id is None:
            id_string = str(self.input) + str(self.expected_output)
            self.id = hashlib.md5(id_string.encode()).hexdigest()
        else:
            self.id = id

    def dict(self):
        return {
            "input": self.input,
            "expected_output": self.expected_output,
            "metrics": [metric.dict() for metric in self.metrics],
            "id": self.id,
        }


class TestCases(UserList):
    """A list of test cases for easy access"""

    @classmethod
    def from_csv(
        self,
        csv_filename: str,
        input_column: str,
        expected_output_column: str,
        id_column: str = None,
        metrics: List[Metric] = None,
    ):
        import pandas as pd

        df = pd.read_csv(csv_filename)
        inputs = df[input_column].values
        expected_outputs = df[expected_output_column].values
        if id_column is not None:
            ids = df[id_column].values
        for i, input in enumerate(inputs):
            self.data.append(
                TestCase(
                    input=input,
                    expected_output=expected_outputs[i],
                    metrics=metrics,
                    id=ids[i],
                )
            )
        return self
