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
    success: bool
    metrics_metadata: List[MetricMetadata] = Field(..., alias="metricsMetadata")
    run_duration: float = Field(..., alias="runDuration")
    latency: Optional[float] = Field(None)
    cost: Optional[float] = Field(None)
    traceStack: Optional[dict] = Field(None)
    context: Optional[list] = Field(None)
    retrieval_context: Optional[list] = Field(None, alias="retrievalContext")
    id: Optional[str] = None
    evaluation_cost: Union[float, None] = Field(None, alias="evaluationCost")

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


class TestRunHttpResponse(BaseModel):
    testRunId: str
    projectId: str
    link: str
