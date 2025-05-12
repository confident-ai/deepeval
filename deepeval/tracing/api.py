from enum import Enum
from typing import Dict, List, Optional, Union, Literal, Any
from pydantic import BaseModel, Field

from deepeval.test_case.llm_test_case import ToolCall


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


class SpanTestCase(BaseModel):
    input: str
    actual_output: str = Field(alias="actualOutput")
    expected_output: Optional[str] = Field(None, lias="expectedOutput")
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
    trace_uuid: str = Field(alias="traceUuid")
    parent_uuid: Optional[str] = Field(None, alias="parentUuid")
    start_time: str = Field(alias="startTime")
    end_time: str = Field(alias="endTime")
    metadata: Optional[Dict[str, Any]] = None
    input: Optional[Union[Dict, list, str]] = None
    output: Optional[Union[Dict, list, str]] = None
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
    span_test_case: Optional[SpanTestCase] = Field(None, alias="spanTestCase")
    metrics: Optional[List[str]] = Field(None, alias="metrics")
    metrics_data: Optional[List[MetricData]] = Field(None, alias="metricsData")

    class Config:
        use_enum_values = True


class TraceApi(BaseModel):
    uuid: str
    base_spans: List[BaseApiSpan] = Field(alias="baseSpans")
    agent_spans: List[BaseApiSpan] = Field(alias="agentSpans")
    llm_spans: List[BaseApiSpan] = Field(alias="llmSpans")
    retriever_spans: List[BaseApiSpan] = Field(alias="retrieverSpans")
    tool_spans: List[BaseApiSpan] = Field(alias="toolSpans")
    start_time: str = Field(alias="startTime")
    end_time: str = Field(alias="endTime")
    metadata: Optional[Dict[str, Any]] = Field(None)
    tags: Optional[List[str]] = Field(None)
    environment: Optional[str] = Field(None)
