from enum import Enum
from typing import Dict, List, Optional, Union, Literal, Any
from pydantic import BaseModel, Field

from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.test_case.llm_test_case import ToolCall
from deepeval.tracing.context import current_trace_context
from deepeval.utils import is_confident
from deepeval.feedback.api import APIFeedback


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
    trace_uuid: str = Field(alias="traceUuid")
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
    span_test_case: Optional[TraceSpanTestCase] = Field(
        None, alias="spanTestCase"
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
    thread_id: Optional[str] = Field(None, alias="threadId")
    user_id: Optional[str] = Field(None, alias="userId")
    input: Optional[Any] = Field(None)
    output: Optional[Any] = Field(None)
    feedback: Optional[APIFeedback] = Field(None)

    # evals
    trace_test_case: Optional[TraceSpanTestCase] = Field(
        None, alias="traceTestCase"
    )
    metric_collection: Optional[str] = Field(None, alias="metricCollection")
    metrics_data: Optional[List[MetricData]] = Field(None, alias="metricsData")


class RunThreadMetricApi(BaseModel):
    thread_supplied_id: str = Field(alias="threadSuppliedId")
    metric_collection: str = Field(alias="metricCollection")


def evaluate_thread(thread_id: str, metric_collection: str):
    trace = current_trace_context.get()
    api_key = None
    if trace:
        api_key = trace.confident_api_key
    if not api_key and not is_confident():
        return

    run_thread_metric_api = RunThreadMetricApi(
        threadSuppliedId=thread_id,
        metricCollection=metric_collection,
    )
    try:
        body = run_thread_metric_api.model_dump(
            by_alias=True,
            exclude_none=True,
        )
    except AttributeError:
        # Pydantic version below 2.0
        body = run_thread_metric_api.dict(by_alias=True, exclude_none=True)

    api = Api(api_key=api_key)
    api.send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.THREAD_METRICS_ENDPOINT,
        body=body,
    )


async def a_evaluate_thread(thread_id: str, metric_collection: str):
    trace = current_trace_context.get()
    api_key = None
    if trace:
        api_key = trace.confident_api_key
    if not api_key and not is_confident():
        return

    run_thread_metric_api = RunThreadMetricApi(
        threadSuppliedId=thread_id,
        metricCollection=metric_collection,
    )
    try:
        body = run_thread_metric_api.model_dump(
            by_alias=True,
            exclude_none=True,
        )
    except AttributeError:
        # Pydantic version below 2.0
        body = run_thread_metric_api.dict(by_alias=True, exclude_none=True)

    api = Api(api_key=api_key)
    await api.a_send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.THREAD_METRICS_ENDPOINT,
        body=body,
    )
