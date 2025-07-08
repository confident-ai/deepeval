from pydantic import BaseModel
from typing import List


class RoleViolationVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[RoleViolationVerdict]


class RoleViolations(BaseModel):
    role_violations: List[str]


class RoleViolationScoreReason(BaseModel):
    reason: str
