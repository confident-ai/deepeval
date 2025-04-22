from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Union, Dict, List

from deepeval.test_run.api import LLMApiTestCase, MetricData

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
    llm_api_test_case: Optional[LLMApiTestCase] = Field(None, alias="llmApiTestCase")

    # success: Union[bool, None] = Field(None)
    # metrics_data: Union[List[MetricData], None] = Field(
    #     None, alias="metricsData"
    # )
    # run_duration: Union[float, None] = Field(None, alias="runDuration")
    # evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")


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

class AgenticApiTestCase(BaseModel):
    name: str
    trace: TraceApi
    # make these optional, not all test cases in a conversation will be evaluated
    success: Union[bool, None] = Field(None)
    metrics_data: Union[List[MetricData], None] = Field(
        None, alias="metricsData"
    )
    run_duration: Union[float, None] = Field(None, alias="runDuration")
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")
    order: Union[int, None] = Field(None)
    # These should map 1 to 1 from golden
    additional_metadata: Optional[Dict] = Field(
        None, alias="additionalMetadata"
    )
    comments: Optional[str] = Field(None)

    class Config:
        use_enum_values = True

    def update_metric_data(self, metric_data: MetricData):
        if self.metrics_data is None:
            self.metrics_data = [metric_data]
        else:
            self.metrics_data.append(metric_data)

        if self.success is None:
            # self.success will be None when it is a message
            # in that case we will be setting success for the first time
            self.success = metric_data.success
        else:
            if metric_data.success is False:
                self.success = False

        evaluationCost = metric_data.evaluation_cost
        if evaluationCost is None:
            return

        if self.evaluation_cost is None:
            self.evaluation_cost = evaluationCost
        else:
            self.evaluation_cost += evaluationCost

    def update_run_duration(self, run_duration: float):
        self.run_duration = run_duration