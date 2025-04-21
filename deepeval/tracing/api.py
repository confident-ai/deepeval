from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Union, Dict, List


class SpanApiType(Enum):
    BASE = "base"
    AGENT = "agent"
    LLM = "llm"
    RETRIEVER = "retriever"
    TOOL = "tool"


class TraceSpanApiStatus(Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"


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
    test_case_input: Optional[str] = Field(None, alias="testCaseInput")
    test_case_actual_output: Optional[str] = Field(
        None, alias="testCaseActualOutput"
    )
    test_case_retrieval_context: Optional[List[str]] = Field(
        None, alias="testCaseRetrievalContext"
    )
    metrics: Optional[List[str]] = Field(None, alias="metrics")

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
