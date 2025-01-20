from pydantic import Field, BaseModel
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class LLMTestCaseParams(Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"
    TOOLS_CALLED = "tools_called"
    EXPECTED_TOOLS = "expected_tools"
    REASONING = "reasoning"

class ToolCall(BaseModel):
    name: str
    description: Optional[str] = None
    reasoning: Optional[str] = None
    output: Optional[Any] = None
    input_parameters: Optional[Dict] = Field(None, alias="evaluationModel")

@dataclass
class LLMTestCase:
    input: str
    actual_output: str
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    additional_metadata: Optional[Dict] = None
    comments: Optional[str] = None
    tools_called: Optional[List[ToolCall]] = None
    expected_tools: Optional[List[ToolCall]] = None
    reasoning: Optional[str] = None
    name: Optional[str] = field(default=None)
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

        # Ensure `tools_called` is None or a list of strings
        if self.tools_called is not None:
            if not isinstance(self.tools_called, list) or not all(
                isinstance(item, ToolCall) for item in self.tools_called
            ):
                raise TypeError(
                    "'tools_called' must be None or a list of `ToolCall`"
                )

        # Ensure `expected_tools` is None or a list of strings
        if self.expected_tools is not None:
            if not isinstance(self.expected_tools, list) or not all(
                isinstance(item, ToolCall) for item in self.expected_tools
            ):
                raise TypeError(
                    "'expected_tools' must be None or a list of `ToolCall`"
                )
