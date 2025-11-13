from typing import List, Optional, Dict, Union
from urllib.parse import urlparse, unquote
from dataclasses import dataclass, field
from enum import Enum
import mimetypes
import base64
import os

from deepeval.test_case import ToolCall


@dataclass
class MLLMImage:
    dataBase64: Optional[str] = None
    mimeType: Optional[str] = None
    url: Optional[str] = None
    local: Optional[bool] = None
    filename: Optional[str] = None

    def __post_init__(self):

        if self.url and self.dataBase64:
            raise ValueError(
                "You cannot provide both 'url' and 'dataBase64' at the same time when creating an MLLMImage."
            )

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
                self.mimeType = (
                    mimetypes.guess_type(path)[0] or "application/octet-stream"
                )
                with open(path, "rb") as f:
                    raw = f.read()
                self.dataBase64 = base64.b64encode(raw).decode("ascii")
            else:
                self.filename = None
                self.mimeType = None
                self.dataBase64 = None

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

    def as_data_uri(self) -> Optional[str]:
        """Return the image as a data URI string, if Base64 data is available."""
        if not self.dataBase64 or not self.mimeType:
            return None
        return f"data:{self.mimeType};base64,{self.dataBase64}"


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
