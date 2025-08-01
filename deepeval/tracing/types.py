from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
from rich.progress import Progress

from deepeval.feedback import Feedback
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.tracing.attributes import (
    AgentAttributes,
    LlmAttributes,
    RetrieverAttributes,
    ToolAttributes,
)
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric


class TraceWorkerStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"


class SpanType(Enum):
    AGENT = "agent"
    LLM = "llm"
    RETRIEVER = "retriever"
    TOOL = "tool"
    GRAPH_EXECUTION = "graph_execution"
    NODE_EXECUTION = "node_execution"
    STATE_TRANSITION = "state_transition"
    CONDITIONAL_ROUTING = "conditional_routing"
    PARALLEL_EXECUTION = "parallel_execution"


class TraceSpanStatus(Enum):
    SUCCESS = "SUCCESS"
    ERRORED = "ERRORED"
    IN_PROGRESS = "IN_PROGRESS"


class BaseSpan(BaseModel):
    uuid: str
    status: TraceSpanStatus
    children: List["BaseSpan"]
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
    feedback: Optional[Feedback] = None

    # Don't serialize these
    progress: Optional[Progress] = Field(None, exclude=True)
    pbar_callback_id: Optional[int] = Field(None, exclude=True)

    class Config:
        arbitrary_types_allowed = True


class AgentSpan(BaseSpan):
    name: str
    available_tools: List[str] = []
    agent_handoffs: List[str] = []
    attributes: Optional[AgentAttributes] = None

    def set_attributes(self, attributes: AgentAttributes):
        self.attributes = attributes


class LlmSpan(BaseSpan):
    model: Optional[str] = None
    attributes: Optional[LlmAttributes] = None
    cost_per_input_token: Optional[float] = Field(
        None, serialization_alias="costPerInputToken"
    )
    cost_per_output_token: Optional[float] = Field(
        None, serialization_alias="costPerOutputToken"
    )

    def set_attributes(self, attributes: LlmAttributes):
        self.attributes = attributes


class RetrieverSpan(BaseSpan):
    embedder: str
    attributes: Optional[RetrieverAttributes] = None

    def set_attributes(self, attributes: RetrieverAttributes):
        self.attributes = attributes


class ToolSpan(BaseSpan):
    name: str  # Required name for ToolSpan
    attributes: Optional[ToolAttributes] = None
    description: Optional[str] = None

    def set_attributes(self, attributes: ToolAttributes):
        self.attributes = attributes


class GraphSpan(BaseSpan):
    """Span for tracking LangGraph execution"""
    graph_config: Optional[Dict[str, Any]] = None
    node_count: Optional[int] = None
    execution_mode: Optional[str] = None  # 'sequential', 'parallel', 'conditional'
    attributes: Optional["GraphAttributes"] = None

    def set_attributes(self, attributes: "GraphAttributes"):
        self.attributes = attributes


class NodeSpan(BaseSpan):
    """Span for tracking individual node execution in LangGraph"""
    node_type: Optional[str] = None  # 'function', 'conditional', 'join', 'end'
    dependencies: Optional[List[str]] = None
    execution_order: Optional[int] = None
    conditional_logic: Optional[str] = None
    parallel_group: Optional[str] = None
    attributes: Optional["NodeAttributes"] = None

    def set_attributes(self, attributes: "NodeAttributes"):
        self.attributes = attributes


class StateTransitionSpan(BaseSpan):
    """Span for tracking state transitions between nodes"""
    from_node: Optional[str] = None
    to_node: Optional[str] = None
    state_snapshot: Optional[Dict[str, Any]] = None
    transition_reason: Optional[str] = None
    routing_decision: Optional[str] = None
    attributes: Optional["StateTransitionAttributes"] = None

    def set_attributes(self, attributes: "StateTransitionAttributes"):
        self.attributes = attributes


class TurnContext(BaseModel):
    retrieval_context: Optional[List[str]] = Field(
        None, serialization_alias="retrievalContext"
    )
    tools_called: Optional[List[ToolCall]] = Field(
        None, serialization_alias="toolsCalled"
    )


class Trace(BaseModel):
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
    feedback: Optional[Feedback] = None

    llm_test_case: Optional[LLMTestCase] = None
    metrics: Optional[List[BaseMetric]] = None
    metric_collection: Optional[str] = None
    turn_context: Optional[TurnContext] = None

    # Don't serialize these
    confident_api_key: Optional[str] = Field(None, exclude=True)

    class Config:
        arbitrary_types_allowed = True
