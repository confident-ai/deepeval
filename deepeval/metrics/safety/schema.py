from typing import List
from pydantic import BaseModel


class SafetyVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[SafetyVerdict]


class Reason(BaseModel):
    reason: str 