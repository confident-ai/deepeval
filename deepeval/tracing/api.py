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
    start_time: float = Field(alias="startTime")
    end_time: Union[float, None] = Field(None, alias="endTime")
    inputText: Optional[str] = None
    input: Optional[Union[Dict, list]] = None
    outputText: Optional[str] = None
    output: Optional[Union[Dict, list]] = None
    error: Optional[str] = None

    # agents
    available_tools: Optional[List[str]] = Field(None, alias="availableTools")
    handoff_agents: Optional[List[str]] = Field(None, alias="handoffAgents")

    # tools
    description: Optional[str] = None

    # retriever
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

    class Config:
        use_enum_values = True


class TraceApi(BaseModel):
    uuid: str
    base_spans: List[BaseApiSpan] = Field(alias="baseSpans")
    agent_spans: List[BaseApiSpan] = Field(alias="agentSpans")
    llm_spans: List[BaseApiSpan] = Field(alias="llmSpans")
    retriever_spans: List[BaseApiSpan] = Field(alias="retrieverSpans")
    tool_spans: List[BaseApiSpan] = Field(alias="toolSpans")
    start_time: float = Field(alias="startTime")
    end_time: Union[float, None] = Field(None, alias="endTime")
