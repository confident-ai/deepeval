"""Investigate test case.
"""
import hashlib
from collections import UserList
from typing import Any, List, Optional

from .metrics.factual_consistency import FactualConsistencyMetric
from .metrics.metric import Metric


class TestCase:
    pass


class LLMTestCase(TestCase):
    def __init__(
        self,
        query: Any,
        expected_output: Any,
        context: str = None,
        metrics: List[Metric] = None,
        output: Optional[str] = None,
        id: str = None,
    ):
        if metrics is None:
            fact_consistency_metric = FactualConsistencyMetric(
                minimum_score=0.3
            )
            self.metrics = [fact_consistency_metric]
        else:
            self.metrics = metrics
        self.query = query
        self.expected_output = expected_output
        self.context = context
        if id is None:
            id_string = str(self.query) + str(self.expected_output)
            self.id = hashlib.md5(id_string.encode()).hexdigest()
        else:
            self.id = id
        if output is None:
            self.output = output

    def dict(self):
        data = {
            "metrics": self.metrics,
            "id": self.id,
        }
        if self.query:
            data["query"] = self.query
        if self.expected_output:
            data["expected_output"] = self.expected_output
        if self.context:
            data["context"] = self.context
        if self.output:
            data["output"] = self.output
        return data


class AgentTestCase(TestCase):
    pass
