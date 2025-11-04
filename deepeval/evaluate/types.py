from typing import Optional, List, Union, Dict
from dataclasses import dataclass
from pydantic import BaseModel

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
    multimodal: Optional[bool] = None
    input: Union[Optional[str], List[Union[str, MLLMImage]]] = None
    actual_output: Union[Optional[str], List[Union[str, MLLMImage]]] = None
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    turns: Optional[List[TurnApi]] = None
    additional_metadata: Optional[Dict] = None


class EvaluationResult(BaseModel):
    test_results: List[TestResult]
    confident_link: Optional[str]
    test_run_id: Optional[str]


class PostExperimentRequest(BaseModel):
    testRuns: List[TestRun]
    name: Optional[str]
