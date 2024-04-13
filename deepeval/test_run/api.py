from pydantic import BaseModel, Field
from typing import Optional, List, Union


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


class APITestCase(BaseModel):
    name: str
    input: str
    actual_output: str = Field(..., alias="actualOutput")
    expected_output: Optional[str] = Field(None, alias="expectedOutput")
    context: Optional[list] = Field(None)
    retrieval_context: Optional[list] = Field(None, alias="retrievalContext")
    # make optional, test cases in a conversation will NOT be evaluated individually
    success: Union[bool, None] = Field(None)
    # make optional, test cases in a conversation will NOT be evaluated individually
    metrics_metadata: List[MetricMetadata] = Field(..., alias="metricsMetadata")
    # make optional, test cases in a conversation will NOT be evaluated individually
    run_duration: float = Field(..., alias="runDuration")
    # make optional, test cases in a conversation will NOT be evaluated individually
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")

    latency: Optional[float] = Field(None)
    cost: Optional[float] = Field(None)
    traceStack: Optional[dict] = Field(None)

    def update(self, metric_metadata: MetricMetadata):
        self.metrics_metadata.append(metric_metadata)
        if metric_metadata.success is False:
            self.success = False

        evaluationCost = metric_metadata.evaluation_cost
        if evaluationCost is None:
            return

        if self.evaluation_cost is None:
            self.evaluation_cost = evaluationCost
        else:
            self.evaluation_cost += evaluationCost


class ConversationalAPITestCase(BaseModel):
    name: str
    messages: List[APITestCase] = Field(default_factory=lambda: [])
    success: bool
    metrics_metadata: List[MetricMetadata] = Field(..., alias="metricsMetadata")
    run_duration: float = Field(..., alias="runDuration")
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")


class TestRunHttpResponse(BaseModel):
    testRunId: str
    projectId: str
    link: str
