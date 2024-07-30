from pydantic import Field
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class LLMTestCaseParams(Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"
    TOOLS_USED = "tools_used"
    EXPECTED_TOOLS = "expected_tools"
    REASONING = "reasoning"


@dataclass
class LLMTestCase:
    input: str
    actual_output: str
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    additional_metadata: Optional[Dict] = None
    comments: Optional[str] = None
    tools_used: Optional[List[str]] = None
    expected_tools: Optional[List[str]] = None
    reasoning: Optional[List[str]] = None
    _dataset_rank: Optional[int] = field(default=None, repr=False)
    _dataset_alias: Optional[str] = field(default=None, repr=False)
    _dataset_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        # Ensure `context` is None or a list of strings
        if self.context is not None:
            if not isinstance(self.context, list) or not all(
                isinstance(item, str) for item in self.context
            ):
                raise TypeError("'context' must be None or a list of strings")

        # Ensure `retrieval_context` is None or a list of strings
        if self.retrieval_context is not None:
            if not isinstance(self.retrieval_context, list) or not all(
                isinstance(item, str) for item in self.retrieval_context
            ):
                raise TypeError(
                    "'retrieval_context' must be None or a list of strings"
                )

        # Ensure `tools_used` is None or a list of strings
        if self.tools_used is not None:
            if not isinstance(self.tools_used, list) or not all(
                isinstance(item, str) for item in self.tools_used
            ):
                raise TypeError(
                    "'tools_used' must be None or a list of strings"
                )

        # Ensure `expected_tools` is None or a list of strings
        if self.expected_tools is not None:
            if not isinstance(self.expected_tools, list) or not all(
                isinstance(item, str) for item in self.expected_tools
            ):
                raise TypeError(
                    "'expected_tools' must be None or a list of strings"
                )

        # Ensure `reasoning` is None or a list of strings
        if self.reasoning is not None:
            if not isinstance(self.reasoning, list) or not all(
                isinstance(item, str) for item in self.reasoning
            ):
                raise TypeError("'reasoning' must be None or a list of strings")
