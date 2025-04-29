from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from deepeval.test_case import LLMTestCase


class SpanApiType(Enum):
    BASE = "base"
    AGENT = "agent"
    LLM = "llm"
    RETRIEVER = "retriever"
    TOOL = "tool"


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


class BaseApiSpan(BaseModel):
    uuid: str
    name: str = None
    status: TraceSpanApiStatus
    type: SpanApiType
    trace_uuid: str = Field(alias="traceUuid")
    parent_uuid: Optional[str] = Field(None, alias="parentUuid")
    start_time: str = Field(alias="startTime")
    end_time: str = Field(alias="endTime")
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
    llm_test_case: Optional[LLMTestCase] = Field(None, alias="llmTestCase")
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
