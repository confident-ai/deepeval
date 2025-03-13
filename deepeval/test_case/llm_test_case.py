from pydantic import Field, BaseModel
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import json


class LLMTestCaseParams(Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"
    TOOLS_CALLED = "tools_called"
    EXPECTED_TOOLS = "expected_tools"


class ToolCallParams(Enum):
    INPUT_PARAMETERS = "input_parameters"
    OUTPUT = "output"


class ToolCall(BaseModel):
    name: str
    description: Optional[str] = None
    reasoning: Optional[str] = None
    output: Optional[Any] = None
    input_parameters: Optional[Dict[str, Any]] = Field(
        None, serialization_alias="inputParameters"
    )

    def __eq__(self, other):
        if not isinstance(other, ToolCall):
            return False
        return (
            self.name == other.name
            and self.input_parameters == other.input_parameters
            and self.output == other.output
        )

    def __hash__(self):
        input_params = (
            self.input_parameters if self.input_parameters is not None else {}
        )
        output_hashable = (
            frozenset(self.output.items())
            if isinstance(self.output, dict)
            else self.output
        )
        return hash(
            (self.name, frozenset(input_params.items()), output_hashable)
        )

    def __repr__(self):
        fields = []

        # Add basic fields
        if self.name:
            fields.append(f'name="{self.name}"')
        if self.description:
            fields.append(f'description="{self.description}"')
        if self.reasoning:
            fields.append(f'reasoning="{self.reasoning}"')

        # Handle nested fields like input_parameters
        if self.input_parameters:
            formatted_input = json.dumps(self.input_parameters, indent=4)
            formatted_input = self._indent_nested_field(
                "input_parameters", formatted_input
            )
            fields.append(formatted_input)

        # Handle nested fields like output
        if isinstance(self.output, dict):
            formatted_output = json.dumps(self.output, indent=4)
            formatted_output = self._indent_nested_field(
                "output", formatted_output
            )
            fields.append(formatted_output)
        elif self.output is not None:
            fields.append(f"output={repr(self.output)}")

        # Combine fields with proper formatting
        fields_str = ",\n    ".join(fields)
        return f"ToolCall(\n    {fields_str}\n)"

    @staticmethod
    def _indent_nested_field(field_name: str, formatted_field: str) -> str:
        """Helper method to indent multi-line fields for better readability."""
        lines = formatted_field.splitlines()
        return f"{field_name}={lines[0]}\n" + "\n".join(
            f"    {line}" for line in lines[1:]
        )


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
