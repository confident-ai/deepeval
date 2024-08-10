from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict


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
    input: str
    actual_output: str = Field(..., alias="actualOutput")
    expected_output: Optional[str] = Field(None, alias="expectedOutput")
    context: Optional[list] = Field(None)
    retrieval_context: Optional[list] = Field(None, alias="retrievalContext")
    tools_used: Optional[list] = Field(None, alias="toolsUsed")
    expected_tools: Optional[list] = Field(None, alias="expectedTools")
    # make optional, not all test cases in a conversation will be evaluated
    success: Union[bool, None] = Field(None)
    # make optional, not all test cases in a conversation will be evaluated
    metrics_data: Union[List[MetricData], None] = Field(
        None, alias="metricsData"
    )
    # make optional, not all test cases in a conversation will be evaluated
    run_duration: Union[float, None] = Field(None, alias="runDuration")
    # make optional, not all test cases in a conversation will be evaluated
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")
    order: Union[int, None] = Field(None)
    # These should map 1 to 1 from golden
    additional_metadata: Optional[Dict] = Field(
        None, alias="additionalMetadata"
    )
    comments: Optional[str] = Field(None)
    traceStack: Optional[dict] = Field(None)
    conversational_instance_id: Optional[int] = Field(None)

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


class ConversationalApiTestCase(BaseModel):
    name: str
    success: bool
    instance_id: Optional[int] = Field(None)
    # metrics_data can be None when we're not evaluating using conversational metrics
    metrics_data: Union[List[MetricData], None] = Field(
        None, alias="metricsData"
    )
    run_duration: float = Field(0.0, alias="runDuration")
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")
    messages: List[LLMApiTestCase] = Field(
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
