from typing import List
from pydantic import BaseModel


class UnauthorizedExecutionVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[UnauthorizedExecutionVerdict]


class Reason(BaseModel):
    reason: str 