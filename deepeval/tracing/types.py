from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional, Union, Literal
from rich.progress import Progress

from deepeval.utils import make_model_config

from deepeval.prompt.prompt import Prompt
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric


class Message(BaseModel):
    role: str
    """To be displayed on the top of the message block."""

    type: Literal["tool_calls", "tool_output", "thinking", "default"] = (
        "default"
    )
    """Decides how the content is rendered."""

    content: Any
    """The content of the message."""


class TraceWorkerStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"


class SpanType(Enum):
    AGENT = "agent"
    LLM = "llm"
    RETRIEVER = "retriever"
    TOOL = "tool"


class TraceSpanStatus(Enum):
    SUCCESS = "SUCCESS"
    ERRORED = "ERRORED"
    IN_PROGRESS = "IN_PROGRESS"


class LlmToolCall(BaseModel):
    name: str
    args: Dict[str, Any]
    id: Optional[str] = None


class LlmOutput(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[LlmToolCall]] = None


class BaseSpan(BaseModel):
    model_config = make_model_config(arbitrary_types_allowed=True)

    uuid: str
    status: TraceSpanStatus
    children: List["BaseSpan"] = Field(default_factory=list)
    trace_uuid: str = Field(serialization_alias="traceUuid")
    parent_uuid: Optional[str] = Field(None, serialization_alias="parentUuid")
    start_time: float = Field(serialization_alias="startTime")
    end_time: Union[float, None] = Field(None, serialization_alias="endTime")
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    error: Optional[str] = None
    llm_test_case: Optional[LLMTestCase] = None
    metrics: Optional[List[BaseMetric]] = None
    metric_collection: Optional[str] = None

    # Don't serialize these
    progress: Optional[Progress] = Field(None, exclude=True)
    pbar_callback_id: Optional[int] = Field(None, exclude=True)

    # additional test case parameters
    retrieval_context: Optional[List[str]] = Field(
        None, serialization_alias="retrievalContext"
    )
    context: Optional[List[str]] = Field(None, serialization_alias="context")
    expected_output: Optional[str] = Field(
        None, serialization_alias="expectedOutput"
    )
    tools_called: Optional[List[ToolCall]] = Field(
        None, serialization_alias="toolsCalled"
    )
    expected_tools: Optional[List[ToolCall]] = Field(
        None, serialization_alias="expectedTools"
    )


class AgentSpan(BaseSpan):
    name: str
    available_tools: List[str] = []
    agent_handoffs: List[str] = []


class LlmSpan(BaseSpan):

    model: Optional[str] = None
    prompt: Optional[Prompt] = None
    input_token_count: Optional[float] = Field(
        None, serialization_alias="inputTokenCount"
    )
    output_token_count: Optional[float] = Field(
        None, serialization_alias="outputTokenCount"
    )
    cost_per_input_token: Optional[float] = Field(
        None, serialization_alias="costPerInputToken"
    )
    cost_per_output_token: Optional[float] = Field(
        None, serialization_alias="costPerOutputToken"
    )
    token_intervals: Optional[Dict[float, str]] = Field(
        None, serialization_alias="tokenTimes"
    )

    # input_tools: Optional[List[ToolSchema]] = Field(None, serialization_alias="inputTools")
    # invocation_params: Optional[Dict[str, Any]] = Field(None, serialization_alias="invocationParams")
    # output_metadata: Optional[Dict[str, Any]] = Field(None, serialization_alias="outputMetadata")

    # for serializing `prompt`
    model_config = make_model_config(arbitrary_types_allowed=True)


class RetrieverSpan(BaseSpan):
    embedder: Optional[str] = None
    top_k: Optional[int] = Field(None, serialization_alias="topK")
    chunk_size: Optional[int] = Field(None, serialization_alias="chunkSize")


class ToolSpan(BaseSpan):
    name: str  # Required name for ToolSpan
    description: Optional[str] = None


class Trace(BaseModel):
    model_config = make_model_config(arbitrary_types_allowed=True)

    uuid: str = Field(serialization_alias="uuid")
    status: TraceSpanStatus
    root_spans: List[BaseSpan] = Field(serialization_alias="rootSpans")
    start_time: float = Field(serialization_alias="startTime")
    end_time: Union[float, None] = Field(None, serialization_alias="endTime")
    name: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    metrics: Optional[List[BaseMetric]] = None
    metric_collection: Optional[str] = None

    # Don't serialize these
    confident_api_key: Optional[str] = Field(None, exclude=True)
    environment: str = Field(None, exclude=True)

    # additional test case parameters
    retrieval_context: Optional[List[str]] = Field(
        None, serialization_alias="retrievalContext"
    )
    context: Optional[List[str]] = Field(None, serialization_alias="context")
    expected_output: Optional[str] = Field(
        None, serialization_alias="expectedOutput"
    )
    tools_called: Optional[List[ToolCall]] = Field(
        None, serialization_alias="toolsCalled"
    )
    expected_tools: Optional[List[ToolCall]] = Field(
        None, serialization_alias="expectedTools"
    )


class TraceAttributes(BaseModel):
    name: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class TestCaseMetricPair:
    test_case: LLMTestCase
    metrics: List[BaseMetric]
    hyperparameters: Optional[Dict[str, Any]] = field(default=None)
