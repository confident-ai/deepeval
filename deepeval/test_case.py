"""Investigate test case.
"""
import hashlib
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class LLMTestCaseParams(Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"


@dataclass
class TestCase:
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            id_string = "".join(str(value) for value in self.__dict__.values())
            self.id = hashlib.md5(id_string.encode()).hexdigest()


@dataclass
class LLMTestCase(TestCase):
    def __init__(
        self,
        input: str,
        actual_output: str,
        expected_output: Optional[str] = None,
        context: Optional[List[str]] = None,
        id: Optional[str] = None,
    ):
        super().__init__(id)
        self.input = input
        self.actual_output = actual_output
        self.expected_output = expected_output
        self.context = context

    def __post_init__(self):
        super().__post_init__()
        self.__name__ = f"LLMTestCase_{self.id}"


class AgentTestCase(TestCase):
    """Test Case For Agents"""

    pass
