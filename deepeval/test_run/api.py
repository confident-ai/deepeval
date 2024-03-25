from pydantic import BaseModel, Field
from typing import Optional, List


class MetricMetadata(BaseModel):
    metric: str
    score: float
    threshold: float
    success: bool
    reason: Optional[str] = None
    strict_mode: Optional[bool] = Field(False, alias="strictMode")
    evaluation_model: Optional[str] = Field(None, alias="evaluationModel")


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


class TestRunHttpResponse(BaseModel):
    testRunId: str
    projectId: str
    link: str
