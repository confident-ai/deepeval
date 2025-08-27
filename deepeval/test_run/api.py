from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Union, Dict

from deepeval.test_case import MLLMImage, ToolCall
from deepeval.tracing.api import TraceApi, MetricData


class LLMApiTestCase(BaseModel):
    name: str
    input: str
    actual_output: Optional[str] = Field(None, alias="actualOutput")
    expected_output: Optional[str] = Field(None, alias="expectedOutput")
    context: Optional[list] = Field(None)
    retrieval_context: Optional[list] = Field(None, alias="retrievalContext")
    tools_called: Optional[list] = Field(None, alias="toolsCalled")
    expected_tools: Optional[list] = Field(None, alias="expectedTools")
    token_cost: Optional[float] = Field(None, alias="tokenCost")
    completion_time: Optional[float] = Field(None, alias="completionTime")
    tags: Optional[List[str]] = Field(None)
    multimodal_input: Optional[List[Union[str, MLLMImage]]] = Field(
        None, alias="multimodalInput"
    )
    multimodal_input_actual_output: Optional[List[Union[str, MLLMImage]]] = (
        Field(None, alias="multimodalActualOutput")
    )
    multimodal_expected_output: Optional[List[Union[str, MLLMImage]]] = Field(
        None, alias="multimodalExpectedOutput"
    )
    multimodal_retrieval_context: Optional[List[Union[str, MLLMImage]]] = Field(
        None, alias="multimodalRetrievalContext"
    )
    multimodal_context: Optional[List[Union[str, MLLMImage]]] = Field(
        None, alias="multimodalContext"
    )

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
    trace: Optional[TraceApi] = Field(None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    def update_status(self, success: bool):
        if self.success is None:
            self.success = success
        else:
            if success is False:
                self.success = False

    def is_multimodal(self):
        if (
            self.multimodal_input is not None
            and self.multimodal_input_actual_output is not None
        ):
            return True

        return False


class TurnApi(BaseModel):
    role: str
    content: str
    order: int
    retrieval_context: Optional[list] = Field(None, alias="retrievalContext")
    tools_called: Optional[List[ToolCall]] = Field(None, alias="toolsCalled")
    additional_metadata: Optional[Dict] = Field(
        None, alias="additionalMetadata"
    )
    comments: Optional[str] = Field(None)


class ConversationalApiTestCase(BaseModel):
    name: str
    success: bool
    metrics_data: List[MetricData] = Field(alias="metricsData")
    run_duration: float = Field(0.0, alias="runDuration")
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")
    turns: List[TurnApi] = Field(default_factory=lambda: [])
    order: Union[int, None] = Field(None)
    scenario: Optional[str] = Field(None)
    expected_outcome: Optional[str] = Field(None, alias="expectedOutcome")
    user_description: Optional[str] = Field(None, alias="userDescription")
    context: Optional[list] = Field(None)
    comments: Optional[str] = Field(None)
    additional_metadata: Optional[Dict] = Field(
        None, alias="additionalMetadata"
    )
    tags: Optional[List[str]] = Field(None)

    def update_metric_data(self, metrics_data: MetricData):
        if self.metrics_data is None:
            self.metrics_data = [metrics_data]
        else:
            self.metrics_data.append(metrics_data)

        if metrics_data.success is False:
            self.success = False

        evaluationCost = metrics_data.evaluation_cost
        if evaluationCost is None:
            return

        if self.evaluation_cost is None:
            self.evaluation_cost = evaluationCost
        else:
            self.evaluation_cost += evaluationCost

    def update_run_duration(self, run_duration: float):
        self.run_duration += run_duration


class TestRunHttpResponse(BaseModel):
    id: str
