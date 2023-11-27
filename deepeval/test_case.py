from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class LLMTestCaseParams(Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"


@dataclass
class LLMTestCase:
    def __init__(
        self,
        input: str,
        actual_output: str,
        expected_output: Optional[str] = None,
        context: Optional[List[str]] = None,
        retrieval_context: Optional[List[str]] = None,
        id: Optional[str] = None,
    ):
        self.id = id
        self.input = input
        self.actual_output = actual_output
        self.expected_output = expected_output
        self.context = context
        self.retrieval_context = retrieval_context
