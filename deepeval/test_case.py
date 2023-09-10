"""Investigate test case.
"""
import hashlib
from collections import UserList
from dataclasses import dataclass
from typing import Any, List, Optional

from .metrics.factual_consistency import FactualConsistencyMetric
from .metrics.metric import Metric


@dataclass
class TestCase:
    pass


@dataclass
class LLMTestCase(TestCase):
    query: Optional[str] = None
    expected_output: Optional[str] = None
    context: Optional[str] = None
    metrics: Optional[List[Metric]] = None
    output: Optional[str] = None
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            id_string = (
                str(self.query) + str(self.expected_output) + str(self.context)
            )
            self.id = hashlib.md5(id_string.encode()).hexdigest()

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
    """Test Case For Agents"""

    pass
