from pydantic import BaseModel
from deepeval.metrics.base_metric import YesNo
from typing import List


class RoleViolationVerdict(BaseModel):
    verdict: YesNo
    reason: str


class Verdicts(BaseModel):
    verdicts: List[RoleViolationVerdict]


class RoleViolations(BaseModel):
    role_violations: List[str]


class RoleViolationScoreReason(BaseModel):
    reason: str
