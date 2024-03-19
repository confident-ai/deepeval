from pydantic import BaseModel, Field
from typing import Optional, List
from deepeval.test_case import LLMTestCaseParams
from deepeval.types import Languages
from langchain_core.embeddings import Embeddings

class MetricsMetadata(BaseModel):
    strict_mode: bool
    metric: str
    score: float
    threshold: float
    success: bool
    evaluation_model: Optional[str] = None
    reason: Optional[str] = None

    # Optional attributes (Not every BaseMetric has these attributes)
    criteria: Optional[str] = None
    include_reason: Optional[bool] = None
    n: Optional[int] = None

    # Below problematic either because
    # gets updated after running metric or
    # Cannot compress into JSON format

    #evaluation_steps: Optional[List[str]] = None
    #assessment_questions: Optional[List[str]] = None
    
    #language: Optional[Languages] = None
    #embeddings: Optional[Embeddings] = None
    #evaluation_params: Optional[List[LLMTestCaseParams]] = None

    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types (Embeddings)


class APITestCase(BaseModel):
    name: str
    input: str
    actual_output: str = Field(..., alias="actualOutput")
    expected_output: Optional[str] = Field(None, alias="expectedOutput")
    success: bool
    metrics_metadata: List[MetricsMetadata] = Field(
        ..., alias="metricsMetadata"
    )
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
