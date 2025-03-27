import os
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
from enum import Enum

from deepeval.test_case import ToolCall


@dataclass
class MLLMImage:
    url: str
    local: Optional[bool] = None

    def __post_init__(self):
        if self.local == None:
            self.local = self.is_local_path(self.url)

    @staticmethod
    def is_local_path(url):
        # Parse the URL
        parsed_url = urlparse(url)

        # Check if it's a file scheme or an empty scheme with a local path
        if parsed_url.scheme == "file" or parsed_url.scheme == "":
            # Check if the path exists on the filesystem
            return os.path.exists(parsed_url.path)

        return False


class MLLMTestCaseParams(Enum):
    INPUT = "input"
    ACTUAL_OUTPUT = "actual_output"
    EXPECTED_OUTPUT = "expected_output"
    CONTEXT = "context"
    RETRIEVAL_CONTEXT = "retrieval_context"
    TOOLS_CALLED = "tools_called"
    EXPECTED_TOOLS = "expected_tools"


@dataclass
class MLLMTestCase:
    input: List[Union[str, MLLMImage]]
    actual_output: List[Union[str, MLLMImage]]
    expected_output: Optional[List[Union[str, MLLMImage]]] = None
    context: Optional[List[Union[str, MLLMImage]]] = None
    retrieval_context: Optional[List[Union[str, MLLMImage]]] = None
    additional_metadata: Optional[Dict] = None
    comments: Optional[str] = None
    tools_called: Optional[List[ToolCall]] = None
    expected_tools: Optional[List[ToolCall]] = None
    token_cost: Optional[float] = None
    completion_time: Optional[float] = None
    name: Optional[str] = field(default=None)
    _dataset_rank: Optional[int] = field(default=None, repr=False)
    _dataset_alias: Optional[str] = field(default=None, repr=False)
    _dataset_id: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        # Ensure `expected_output` is None or a list of strings or MLLMImage instances
        if self.expected_output is not None:
            if not isinstance(self.expected_output, list) or not all(
                isinstance(item, (str, MLLMImage))
                for item in self.expected_output
            ):
                raise TypeError(
                    "'expected_output' must be None or a list of strings or MLLMImage instances"
                )

        # Ensure `context` is None or a list of strings or MLLMImage instances
        if self.context is not None:
            if not isinstance(self.context, list) or not all(
                isinstance(item, (str, MLLMImage)) for item in self.context
            ):
                raise TypeError(
                    "'context' must be None or a list of strings or MLLMImage instances"
                )

        # Ensure `retrieval_context` is None or a list of strings or MLLMImage instances
        if self.retrieval_context is not None:
            if not isinstance(self.retrieval_context, list) or not all(
                isinstance(item, (str, MLLMImage))
                for item in self.retrieval_context
            ):
                raise TypeError(
                    "'retrieval_context' must be None or a list of strings or MLLMImage instances"
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
