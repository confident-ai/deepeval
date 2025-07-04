from pydantic import BaseModel, Field
from typing import List


class RoleViolationVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[RoleViolationVerdict]


class Reason(BaseModel):
    reason: str 