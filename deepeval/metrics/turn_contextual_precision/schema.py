from typing import List, Optional
from pydantic import BaseModel


class ContextualPrecisionVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[ContextualPrecisionVerdict]


class ContextualPrecisionScoreReason(BaseModel):
    reason: str


class InteractionContextualPrecisionScore(BaseModel):
    score: float
    reason: Optional[str]
    verdicts: Optional[List[ContextualPrecisionVerdict]]
