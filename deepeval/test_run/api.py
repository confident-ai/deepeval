from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Any, Optional, List, Union, Dict

from deepeval.test_case import MLLMImage


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


class LLMApiTestCase(BaseModel):
    name: str
    input: Optional[str] = None
    actual_output: Optional[str] = Field(None, alias="actualOutput")
    expected_output: Optional[str] = Field(None, alias="expectedOutput")
    context: Optional[list] = Field(None)
    retrieval_context: Optional[list] = Field(None, alias="retrievalContext")

    multimodal_input: Optional[List[Union[str, MLLMImage]]] = Field(
        None, alias="multimodalInput"
    )
    multimodal_input_actual_output: Optional[List[Union[str, MLLMImage]]] = (
        Field(None, alias="multimodalActualOutput")
    )

    tools_called: Optional[list] = Field(None, alias="toolsCalled")
    expected_tools: Optional[list] = Field(None, alias="expectedTools")

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
    traceStack: Optional[dict] = Field(None)
    conversational_instance_id: Optional[int] = Field(None)

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

    @model_validator(mode="before")
    def check_input_and_multimodal_input(cls, values: Dict[str, Any]):
        input = values.get("input")
        actual_output = values.get("actualOutput")
        multimodal_input = values.get("multimodalInput")
        multimodal_actual_output = values.get("multimodalActualOutput")

        # Ensure that either input/actual_output or multimodal_input/multimodal_actual_output is present
        if (input is None or actual_output is None) and (
            multimodal_input is None or multimodal_actual_output is None
        ):
            raise ValueError(
                "Either 'input' and 'actualOutput' or 'multimodalInput' and 'multimodalActualOutput' must be provided."
            )

        return values

    def is_multimodal(self):
        if (
            self.multimodal_input is not None
            and self.multimodal_input_actual_output is not None
        ):
            return True

        return False


class ConversationalApiTestCase(BaseModel):
    name: str
    success: bool
    instance_id: Optional[int] = Field(None)
    # metrics_data can never be None
    metrics_data: List[MetricData] = Field(alias="metricsData")
    run_duration: float = Field(0.0, alias="runDuration")
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")
    turns: List[LLMApiTestCase] = Field(
        default_factory=lambda: [], alias="testCases"
    )
    order: Union[int, None] = Field(None)

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
    testRunId: str
    projectId: str
    link: str
