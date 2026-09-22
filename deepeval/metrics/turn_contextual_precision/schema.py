from typing import List, Optional
from pydantic import BaseModel
from deepeval.metrics.base_metric import YesNo


class ContextualPrecisionVerdict(BaseModel):
    verdict: YesNo
    reason: str


class Verdicts(BaseModel):
    verdicts: List[ContextualPrecisionVerdict]


class ContextualPrecisionScoreReason(BaseModel):
    reason: str


class InteractionContextualPrecisionScore(BaseModel):
    score: float
    reason: Optional[str]
    verdicts: Optional[List[ContextualPrecisionVerdict]]
