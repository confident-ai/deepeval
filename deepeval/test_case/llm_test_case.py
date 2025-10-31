from pydantic import (
    Field,
    BaseModel,
    model_validator,
    PrivateAttr,
    AliasChoices,
)
from typing import List, Optional, Dict, Any
from enum import Enum
import json
import uuid

from deepeval.utils import make_model_config

from deepeval.test_case.mcp import (
    MCPServer,
    MCPPromptCall,
    MCPResourceCall,
    MCPToolCall,
    validate_mcp_servers,
)


class LLMTestCaseParams(Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"
    TOOLS_CALLED = "tools_called"
    EXPECTED_TOOLS = "expected_tools"
    MCP_SERVERS = "mcp_servers"
    MCP_TOOLS_CALLED = "mcp_tools_called"
    MCP_RESOURCES_CALLED = "mcp_resources_called"
    MCP_PROMPTS_CALLED = "mcp_prompts_called"


class ToolCallParams(Enum):
    INPUT_PARAMETERS = "input_parameters"
    OUTPUT = "output"


def _make_hashable(obj):
    """
    Convert an object to a hashable representation recursively.

    Args:
        obj: The object to make hashable

    Returns:
        A hashable representation of the object
    """
    if obj is None:
        return None
    elif isinstance(obj, dict):
        # Convert dict to tuple of sorted key-value pairs
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (list, tuple)):
        # Convert list/tuple to tuple of hashable elements
        return tuple(_make_hashable(item) for item in obj)
    elif isinstance(obj, set):
        # Convert set to frozenset of hashable elements
        return frozenset(_make_hashable(item) for item in obj)
    elif isinstance(obj, frozenset):
        # Handle frozenset that might contain unhashable elements
        return frozenset(_make_hashable(item) for item in obj)
    else:
        # For primitive hashable types (str, int, float, bool, etc.)
        return obj


class ToolCall(BaseModel):
    name: str
    description: Optional[str] = None
    reasoning: Optional[str] = None
    output: Optional[Any] = None
    input_parameters: Optional[Dict[str, Any]] = Field(
        None,
        serialization_alias="inputParameters",
        validation_alias=AliasChoices("inputParameters", "input_parameters"),
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
        """
        Generate a hash for the ToolCall instance.

        This method handles complex input parameters and outputs that may contain
        unhashable types like lists, dicts, and nested structures.

        Returns:
            int: Hash value for this ToolCall instance
        """
        # Handle input_parameters
        input_params = (
            self.input_parameters if self.input_parameters is not None else {}
        )
        input_params_hashable = _make_hashable(input_params)

        # Handle output - use the new helper function instead of manual handling
        output_hashable = _make_hashable(self.output)

        return hash((self.name, input_params_hashable, output_hashable))

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
            formatted_input = json.dumps(
                self.input_parameters, indent=4, ensure_ascii=False
            )
            formatted_input = self._indent_nested_field(
                "input_parameters", formatted_input
            )
            fields.append(formatted_input)

        # Handle nested fields like output
        if isinstance(self.output, dict):
            formatted_output = json.dumps(
                self.output, indent=4, ensure_ascii=False
            )
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


class LLMTestCase(BaseModel):
    model_config = make_model_config(extra="ignore")

    input: str
    actual_output: Optional[str] = Field(
        default=None,
        serialization_alias="actualOutput",
        validation_alias=AliasChoices("actualOutput", "actual_output"),
    )
    expected_output: Optional[str] = Field(
        default=None,
        serialization_alias="expectedOutput",
        validation_alias=AliasChoices("expectedOutput", "expected_output"),
    )
    context: Optional[List[str]] = Field(
        default=None, serialization_alias="context"
    )
    retrieval_context: Optional[List[str]] = Field(
        default=None,
        serialization_alias="retrievalContext",
        validation_alias=AliasChoices("retrievalContext", "retrieval_context"),
    )
    additional_metadata: Optional[Dict] = Field(
        default=None,
        serialization_alias="additionalMetadata",
        validation_alias=AliasChoices(
            "additionalMetadata", "additional_metadata"
        ),
    )
    tools_called: Optional[List[ToolCall]] = Field(
        default=None,
        serialization_alias="toolsCalled",
        validation_alias=AliasChoices("toolsCalled", "tools_called"),
    )
    comments: Optional[str] = Field(
        default=None, serialization_alias="comments"
    )
    expected_tools: Optional[List[ToolCall]] = Field(
        default=None,
        serialization_alias="expectedTools",
        validation_alias=AliasChoices("expectedTools", "expected_tools"),
    )
    token_cost: Optional[float] = Field(
        default=None,
        serialization_alias="tokenCost",
        validation_alias=AliasChoices("tokenCost", "token_cost"),
    )
    completion_time: Optional[float] = Field(
        default=None,
        serialization_alias="completionTime",
        validation_alias=AliasChoices("completionTime", "completion_time"),
    )
    name: Optional[str] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)
    mcp_servers: Optional[List[MCPServer]] = Field(default=None)
    mcp_tools_called: Optional[List[MCPToolCall]] = Field(
        default=None,
        serialization_alias="mcpToolsCalled",
    )
    mcp_resources_called: Optional[List[MCPResourceCall]] = Field(
        default=None, serialization_alias="mcpResourcesCalled"
    )
    mcp_prompts_called: Optional[List[MCPPromptCall]] = Field(
        default=None, serialization_alias="mcpPromptsCalled"
    )
    _trace_dict: Optional[Dict] = PrivateAttr(default=None)
    _dataset_rank: Optional[int] = PrivateAttr(default=None)
    _dataset_alias: Optional[str] = PrivateAttr(default=None)
    _dataset_id: Optional[str] = PrivateAttr(default=None)
    _identifier: Optional[str] = PrivateAttr(
        default_factory=lambda: str(uuid.uuid4())
    )

    @model_validator(mode="before")
    def validate_input(cls, data):
        input = data.get("input")
        actual_output = data.get("actual_output")
        context = data.get("context")
        retrieval_context = data.get("retrieval_context")
        tools_called = data.get("tools_called")
        expected_tools = data.get("expected_tools")
        mcp_servers = data.get("mcp_servers")
        mcp_tools_called = data.get("mcp_tools_called")
        mcp_resources_called = data.get("mcp_resources_called")
        mcp_prompts_called = data.get("mcp_prompts_called")

        if input is not None:
            if not isinstance(input, str):
                raise TypeError("'input' must be a string")

        if actual_output is not None:
            if not isinstance(actual_output, str):
                raise TypeError("'actual_output' must be a string")

        # Ensure `context` is None or a list of strings
        if context is not None:
            if not isinstance(context, list) or not all(
                isinstance(item, str) for item in context
            ):
                raise TypeError("'context' must be None or a list of strings")

        # Ensure `retrieval_context` is None or a list of strings
        if retrieval_context is not None:
            if not isinstance(retrieval_context, list) or not all(
                isinstance(item, str) for item in retrieval_context
            ):
                raise TypeError(
                    "'retrieval_context' must be None or a list of strings"
                )

        # Ensure `tools_called` is None or a list of strings
        if tools_called is not None:
            if not isinstance(tools_called, list) or not all(
                isinstance(item, ToolCall) for item in tools_called
            ):
                raise TypeError(
                    "'tools_called' must be None or a list of `ToolCall`"
                )

        # Ensure `expected_tools` is None or a list of strings
        if expected_tools is not None:
            if not isinstance(expected_tools, list) or not all(
                isinstance(item, ToolCall) for item in expected_tools
            ):
                raise TypeError(
                    "'expected_tools' must be None or a list of `ToolCall`"
                )

        # Ensure `mcp_server` is None or a list of `MCPServer`
        if mcp_servers is not None:
            if not isinstance(mcp_servers, list) or not all(
                isinstance(item, MCPServer) for item in mcp_servers
            ):
                raise TypeError(
                    "'mcp_server' must be None or a list of 'MCPServer'"
                )
            else:
                validate_mcp_servers(mcp_servers)

        # Ensure `mcp_tools_called` is None or a list of `MCPToolCall`
        if mcp_tools_called is not None:
            from mcp.types import CallToolResult

            if not isinstance(mcp_tools_called, list) or not all(
                isinstance(tool_called, MCPToolCall)
                and isinstance(tool_called.result, CallToolResult)
                for tool_called in mcp_tools_called
            ):
                raise TypeError(
                    "The 'tools_called' must be a list of 'MCPToolCall' with result of type 'CallToolResult' from mcp.types"
                )

        # Ensure `mcp_resources_called` is None or a list of `MCPResourceCall`
        if mcp_resources_called is not None:
            from mcp.types import ReadResourceResult

            if not isinstance(mcp_resources_called, list) or not all(
                isinstance(resource_called, MCPResourceCall)
                and isinstance(resource_called.result, ReadResourceResult)
                for resource_called in mcp_resources_called
            ):
                raise TypeError(
                    "The 'resources_called' must be a list of 'MCPResourceCall' with result of type 'ReadResourceResult' from mcp.types"
                )

        # Ensure `mcp_prompts_called` is None or a list of `MCPPromptCall`
        if mcp_prompts_called is not None:
            from mcp.types import GetPromptResult

            if not isinstance(mcp_prompts_called, list) or not all(
                isinstance(prompt_called, MCPPromptCall)
                and isinstance(prompt_called.result, GetPromptResult)
                for prompt_called in mcp_prompts_called
            ):
                raise TypeError(
                    "The 'prompts_called' must be a list of 'MCPPromptCall' with result of type 'GetPromptResult' from mcp.types"
                )

        return data
