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
import re
import os
import mimetypes
import base64
from dataclasses import dataclass, field
from urllib.parse import urlparse, unquote
from deepeval.utils import make_model_config

from deepeval.test_case.mcp import (
    MCPServer,
    MCPPromptCall,
    MCPResourceCall,
    MCPToolCall,
    validate_mcp_servers,
)

_MLLM_IMAGE_REGISTRY: Dict[str, "MLLMImage"] = {}


@dataclass
class MLLMImage:
    dataBase64: Optional[str] = None
    mimeType: Optional[str] = None
    url: Optional[str] = None
    local: Optional[bool] = None
    filename: Optional[str] = None
    _id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self):

        if not self.url and not self.dataBase64:
            raise ValueError(
                "You must provide either a 'url' or both 'dataBase64' and 'mimeType' to create an MLLMImage."
            )

        if self.dataBase64 is not None:
            if self.mimeType is None:
                raise ValueError(
                    "mimeType must be provided when initializing from Base64 data."
                )
        else:
            is_local = self.is_local_path(self.url)
            if self.local is not None:
                assert self.local == is_local, "Local path mismatch"
            else:
                self.local = is_local

            # compute filename, mime_type, and Base64 data
            if self.local:
                path = self.process_url(self.url)
                self.filename = os.path.basename(path)
                self.mimeType = mimetypes.guess_type(path)[0] or "image/jpeg"

                if not os.path.exists(path):
                    raise FileNotFoundError(f"Image file not found: {path}")

                self._load_base64(path)
            else:
                if not self.url.startswith(("http://", "https://")):
                    raise ValueError(
                        f"Invalid remote URL format: {self.url}. URL must start with http:// or https://"
                    )
                self.filename = None
                self.mimeType = None
                self.dataBase64 = None

        _MLLM_IMAGE_REGISTRY[self._id] = self

    def _load_base64(self, path: str):
        with open(path, "rb") as f:
            raw = f.read()
        self.dataBase64 = base64.b64encode(raw).decode("ascii")

    def ensure_images_loaded(self):
        if self.local and self.dataBase64 is None:
            path = self.process_url(self.url)
            self._load_base64(path)
        return self

    def _placeholder(self) -> str:
        return f"[DEEPEVAL:IMAGE:{self._id}]"

    def __str__(self) -> str:
        return self._placeholder()

    def __repr__(self) -> str:
        return self._placeholder()

    def __format__(self, format_spec: str) -> str:
        return self._placeholder()

    @staticmethod
    def process_url(url: str) -> str:
        if os.path.exists(url):
            return url
        parsed = urlparse(url)
        if parsed.scheme == "file":
            raw_path = (
                f"//{parsed.netloc}{parsed.path}"
                if parsed.netloc
                else parsed.path
            )
            path = unquote(raw_path)
            return path
        return url

    @staticmethod
    def is_local_path(url: str) -> bool:
        if os.path.exists(url):
            return True
        parsed = urlparse(url)
        if parsed.scheme == "file":
            raw_path = (
                f"//{parsed.netloc}{parsed.path}"
                if parsed.netloc
                else parsed.path
            )
            path = unquote(raw_path)
            return os.path.exists(path)
        return False

    def parse_multimodal_string(s: str):
        pattern = r"\[DEEPEVAL:IMAGE:(.*?)\]"
        matches = list(re.finditer(pattern, s))

        result = []
        last_end = 0

        for m in matches:
            start, end = m.span()

            if start > last_end:
                result.append(s[last_end:start])

            img_id = m.group(1)

            if img_id not in _MLLM_IMAGE_REGISTRY:
                MLLMImage(url=img_id, _id=img_id)

            result.append(_MLLM_IMAGE_REGISTRY[img_id])
            last_end = end

        if last_end < len(s):
            result.append(s[last_end:])

        return result

    def as_data_uri(self) -> Optional[str]:
        """Return the image as a data URI string, if Base64 data is available."""
        if not self.dataBase64 or not self.mimeType:
            return None
        return f"data:{self.mimeType};base64,{self.dataBase64}"


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
    multimodal: bool = Field(default=False)
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

    @model_validator(mode="after")
    def set_is_multimodal(self):
        import re

        if self.multimodal is True:
            return self

        pattern = r"\[DEEPEVAL:IMAGE:(.*?)\]"

        auto_detect = (
            any(
                [
                    re.search(pattern, self.input or "") is not None,
                    re.search(pattern, self.actual_output or "") is not None,
                    re.search(pattern, self.expected_output or "") is not None,
                ]
            )
            if isinstance(self.input, str)
            else self.multimodal
        )
        if self.retrieval_context is not None:
            auto_detect = auto_detect or any(
                re.search(pattern, context) is not None
                for context in self.retrieval_context
            )
        if self.context is not None:
            auto_detect = auto_detect or any(
                re.search(pattern, context) is not None
                for context in self.context
            )

        self.multimodal = auto_detect
        return self

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

    def _get_images_mapping(self) -> Dict[str, MLLMImage]:
        pattern = r"\[DEEPEVAL:IMAGE:(.*?)\]"
        image_ids = set()

        def extract_ids_from_string(s: Optional[str]) -> None:
            """Helper to extract image IDs from a string."""
            if s is not None and isinstance(s, str):
                matches = re.findall(pattern, s)
                image_ids.update(matches)

        def extract_ids_from_list(lst: Optional[List[str]]) -> None:
            """Helper to extract image IDs from a list of strings."""
            if lst is not None:
                for item in lst:
                    extract_ids_from_string(item)

        extract_ids_from_string(self.input)
        extract_ids_from_string(self.actual_output)
        extract_ids_from_string(self.expected_output)
        extract_ids_from_list(self.context)
        extract_ids_from_list(self.retrieval_context)

        images_mapping = {}
        for img_id in image_ids:
            if img_id in _MLLM_IMAGE_REGISTRY:
                images_mapping[img_id] = _MLLM_IMAGE_REGISTRY[img_id]

        return images_mapping if len(images_mapping) > 0 else None
