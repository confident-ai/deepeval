from typing import List
from pydantic import BaseModel


class ContextualPrecisionVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[ContextualPrecisionVerdict]


class ContextualPrecisionScoreReason(BaseModel):
    reason: str
