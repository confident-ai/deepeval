"""Investigate test case.
"""
import hashlib
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class TestCase:
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            id_string = "".join(str(value) for value in self.__dict__.values())
            self.id = hashlib.md5(id_string.encode()).hexdigest()


@dataclass
class LLMTestCase(TestCase):
    query: str = "-"
    expected_output: str = "-"
    context: str = "-"
    output: str = "-"

    def __post_init__(self):
        super().__post_init__()
        self.__name__ = f"LLMTestCase_{self.id}"

    # def dict(self):
    #     data = {
    #         "metrics": self.metrics,
    #         "id": self.id,
    #     }
    #     if self.query:
    #         data["query"] = self.query
    #     if self.expected_output:
    #         data["expected_output"] = self.expected_output
    #     if self.context:
    #         data["context"] = self.context
    #     if self.output:
    #         data["output"] = self.output
    #     return data


@dataclass
class SearchTestCase(TestCase):
    def __init__(
        self,
        output_list: List[Any],
        golden_list: List[Any],
        query: Optional[str] = None,
        id: Optional[str] = None,
    ):
        super().__init__(id)
        self.output_list = output_list
        self.golden_list = golden_list
        self.query = query


class AgentTestCase(TestCase):
    """Test Case For Agents"""

    pass
