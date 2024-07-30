from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict


class MetricMetadata(BaseModel):
    metric: str
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
    metrics_metadata: Union[List[MetricMetadata], None] = Field(
        None, alias="metricsMetadata"
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

    def update(self, metric_metadata: MetricMetadata):
        if self.metrics_metadata is None:
            self.metrics_metadata = [metric_metadata]
        else:
            self.metrics_metadata.append(metric_metadata)

        if self.success is None:
            # self.success will be None when it is a message
            # in that case we will be setting success for the first time
            self.success = metric_metadata.success
        else:
            if metric_metadata.success is False:
                self.success = False

        evaluationCost = metric_metadata.evaluation_cost
        if evaluationCost is None:
            return

        if self.evaluation_cost is None:
            self.evaluation_cost = evaluationCost
        else:
            self.evaluation_cost += evaluationCost


class ConversationalApiTestCase(BaseModel):
    name: str
    success: bool
    # metrics_metadata can be None when we're not evaluating using conversational metrics
    metrics_metadata: Union[List[MetricMetadata], None] = Field(
        None, alias="metricsMetadata"
    )
    run_duration: float = Field(0.0, alias="runDuration")
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")
    messages: List[LLMApiTestCase] = Field(
        default_factory=lambda: [], alias="testCases"
    )
    order: Union[int, None] = Field(None)

    def update(self, metric_metadata: MetricMetadata, index: int):
        if index == -1:
            pass
        else:
            # if index != -1, update the metrics metadata of the specific message (llm test case)
            self.messages[index].update(metric_metadata)
            if metric_metadata.success is False:
                self.success = False

            evaluationCost = metric_metadata.evaluation_cost
            if evaluationCost is None:
                return

            if self.evaluation_cost is None:
                self.evaluation_cost = evaluationCost
            else:
                self.evaluation_cost += evaluationCost


class TestRunHttpResponse(BaseModel):
    testRunId: str
    projectId: str
    link: str
