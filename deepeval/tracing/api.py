from enum import Enum
from typing import Dict, List, Optional, Union, Literal, Any
from pydantic import BaseModel, Field

from deepeval.feedback.api import APIFeedback
from deepeval.test_case import ToolCall
from deepeval.tracing.types import TurnContext


class SpanApiType(Enum):
    BASE = "base"
    AGENT = "agent"
    LLM = "llm"
    RETRIEVER = "retriever"
    TOOL = "tool"


span_api_type_literals = Literal["base", "agent", "llm", "retriever", "tool"]


class TraceSpanApiStatus(Enum):
    SUCCESS = "SUCCESS"
    ERRORED = "ERRORED"


class MetricData(BaseModel):
    name: str
    threshold: float
    success: bool
    score: Optional[float] = None
    reason: Optional[str] = None
    strict_mode: Optional[bool] = Field(False, alias="strictMode")
    evaluation_model: Optional[str] = Field(None, alias="evaluationModel")
    error: Optional[str] = None
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")
    verbose_logs: Optional[str] = Field(None, alias="verboseLogs")
    standard_deviation: Optional[float] = Field(None, alias="standardDeviation")
    repeat: Optional[int] = Field(None)

class TraceSpanTestCase(BaseModel):
    input: str
    actual_output: str = Field(alias="actualOutput")
    expected_output: Optional[str] = Field(None, alias="expectedOutput")
    retrieval_context: Optional[List[str]] = Field(
        None, alias="retrievalContext"
    )
    context: Optional[List[str]] = Field(None, alias="context")
    tools_called: Optional[List[ToolCall]] = Field(None, alias="toolsCalled")
    expected_tools: Optional[List[ToolCall]] = Field(
        None, alias="expectedTools"
    )


class BaseApiSpan(BaseModel):
    uuid: str
    name: str = None
    status: TraceSpanApiStatus
    type: SpanApiType
    parent_uuid: Optional[str] = Field(None, alias="parentUuid")
    start_time: str = Field(alias="startTime")
    end_time: str = Field(alias="endTime")
    metadata: Optional[Dict[str, Any]] = None
    input: Optional[Any] = Field(None)
    output: Optional[Any] = Field(None)
    error: Optional[str] = None

    # agents
    available_tools: Optional[List[str]] = Field(None, alias="availableTools")
    agent_handoffs: Optional[List[str]] = Field(None, alias="agentHandoffs")

    # tools
    description: Optional[str] = None

    # retriever
    embedder: Optional[str] = None
    top_k: Optional[int] = Field(None, alias="topK")
    chunk_size: Optional[int] = Field(None, alias="chunkSize")

    # llm
    model: Optional[str] = None
    input_token_count: Optional[int] = Field(None, alias="inputTokenCount")
    output_token_count: Optional[int] = Field(None, alias="outputTokenCount")
    cost_per_input_token: Optional[float] = Field(
        None, alias="costPerInputToken"
    )
    cost_per_output_token: Optional[float] = Field(
        None, alias="costPerOutputToken"
    )

    ## evals
    llm_test_case: Optional[TraceSpanTestCase] = Field(
        None, alias="llmTestCase"
    )
    metric_collection: Optional[str] = Field(None, alias="metricCollection")
    metrics_data: Optional[List[MetricData]] = Field(None, alias="metricsData")

    ## human feedback
    feedback: Optional[APIFeedback] = Field(None)

    class Config:
        use_enum_values = True
        validate_assignment = True


class TraceApi(BaseModel):
    uuid: str
    base_spans: Optional[List[BaseApiSpan]] = Field(None, alias="baseSpans")
    agent_spans: Optional[List[BaseApiSpan]] = Field(None, alias="agentSpans")
    llm_spans: Optional[List[BaseApiSpan]] = Field(None, alias="llmSpans")
    retriever_spans: Optional[List[BaseApiSpan]] = Field(
        None, alias="retrieverSpans"
    )
    tool_spans: Optional[List[BaseApiSpan]] = Field(None, alias="toolSpans")
    start_time: str = Field(alias="startTime")
    end_time: str = Field(alias="endTime")
    name: Optional[str] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)
    tags: Optional[List[str]] = Field(None)
    environment: Optional[str] = Field(None)
    thread_id: Optional[str] = Field(None, alias="threadId")
    user_id: Optional[str] = Field(None, alias="userId")
    input: Optional[Any] = Field(None)
    output: Optional[Any] = Field(None)
    feedback: Optional[APIFeedback] = Field(None)

    # evals
    llm_test_case: Optional[TraceSpanTestCase] = Field(
        None, alias="llmTestCase"
    )
    metric_collection: Optional[str] = Field(None, alias="metricCollection")
    metrics_data: Optional[List[MetricData]] = Field(None, alias="metricsData")
    turn_context: Optional[TurnContext] = Field(None, alias="turnContext")

    # Don't serialize these
    confident_api_key: Optional[str] = Field(None, exclude=True)
