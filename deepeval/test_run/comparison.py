from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class ComparisonReason(str, Enum):
    NO_OFFICIAL_RUN = "NO_OFFICIAL_RUN"
    INCOMPATIBLE_RUNS = "INCOMPATIBLE_RUNS"
    REGRESSION_DETECTED = "REGRESSION_DETECTED"


class MetricComparisonResult(BaseModel):
    metric: str
    official_score: float
    new_score: float
    delta: float
    passed: bool


class TestCaseComparisonResult(BaseModel):
    name: str
    passed: bool
    metrics: List[MetricComparisonResult] = []


class TestRunComparisonResult(BaseModel):
    passed: bool
    reason: Optional[ComparisonReason] = None
    official_run_id: Optional[str] = None
    new_run_id: str
    total_test_cases: int = 0
    regressed_test_cases: int = 0
    test_cases: List[TestCaseComparisonResult] = []
