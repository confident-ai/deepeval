from typing import List, Optional
from pydantic import BaseModel
from deepeval.metrics.base_metric import YesNo


class ContextualRecallVerdict(BaseModel):
    verdict: YesNo
    reason: str


class Verdicts(BaseModel):
    verdicts: List[ContextualRecallVerdict]


class ContextualRecallScoreReason(BaseModel):
    reason: str


class InteractionContextualRecallScore(BaseModel):
    score: float
    reason: Optional[str]
    verdicts: Optional[List[ContextualRecallVerdict]]
