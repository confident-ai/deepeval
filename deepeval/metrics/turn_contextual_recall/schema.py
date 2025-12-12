from typing import List, Optional
from pydantic import BaseModel


class ContextualRecallVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[ContextualRecallVerdict]


class ContextualRecallScoreReason(BaseModel):
    reason: str


class InteractionContextualRecallScore(BaseModel):
    score: float
    reason: Optional[str]
    verdicts: Optional[List[ContextualRecallVerdict]]
