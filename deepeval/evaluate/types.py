from typing import Optional, List, Union, Dict, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field

from deepeval.test_run.api import MetricData, TurnApi
from deepeval.test_case import MLLMImage
from deepeval.test_run import TestRun


@dataclass
class TestResult:
    """Returned from run_test"""

    __test__ = False
    name: str
    success: bool
    metrics_data: Union[List[MetricData], None]
    conversational: bool
    index: Optional[int] = None
    multimodal: Optional[bool] = None
    input: Union[Optional[str], List[Union[str, MLLMImage]]] = None
    actual_output: Union[Optional[str], List[Union[str, MLLMImage]]] = None
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    turns: Optional[List[TurnApi]] = None
    metadata: Optional[Dict] = None


class EvaluationResult(BaseModel):
    test_results: List[TestResult]
    confident_link: Optional[str]
    test_run_id: Optional[str]


class PostExperimentRequest(BaseModel):
    testRuns: List[TestRun]
    name: Optional[str]


class MetricDiff(BaseModel):
    name: str
    old_score: Optional[float] = None
    new_score: Optional[float] = None
    diff: Optional[float] = None
    old_success: Optional[bool] = None
    new_success: Optional[bool] = None
    old_reason: Optional[str] = None
    new_reason: Optional[str] = None
    old_error: Optional[str] = None
    new_error: Optional[str] = None


class LLMTestCaseDiff(BaseModel):
    name: str
    input: Optional[str] = None
    is_conversational: bool = False
    
    old_success: Optional[bool] = None
    new_success: Optional[bool] = None
    
    old_cost: Optional[float] = None
    new_cost: Optional[float] = None
    
    old_latency: Optional[float] = None
    new_latency: Optional[float] = None
    
    metrics: Dict[str, MetricDiff] = Field(default_factory=dict)
    
    # Change status: "improved", "degraded", "unchanged", "added", "removed"
    change_status: str = "unchanged"


class RunComparisonResult(BaseModel):
    run_a_name: str
    run_b_name: str
    
    old_passed: int = 0
    new_passed: int = 0
    old_failed: int = 0
    new_failed: int = 0
    
    old_duration: float = 0.0
    new_duration: float = 0.0
    
    old_cost: Optional[float] = None
    new_cost: Optional[float] = None
    
    # metric_name -> (old_avg, new_avg)
    metric_summaries: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    
    case_diffs: List[LLMTestCaseDiff] = Field(default_factory=list)
