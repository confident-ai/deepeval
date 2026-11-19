from typing import List, Optional
from pydantic import BaseModel, Field
from deepeval.metrics.base_metric import YesNoBorderline


class FaithfulnessVerdict(BaseModel):
    reason: Optional[str] = Field(default=None)
    verdict: YesNoBorderline


class Verdicts(BaseModel):
    verdicts: List[FaithfulnessVerdict]


class Truths(BaseModel):
    truths: List[str]


class Claims(BaseModel):
    claims: List[str]


class FaithfulnessScoreReason(BaseModel):
    reason: str


class InteractionFaithfulnessScore(BaseModel):
    score: float
    reason: Optional[str]
    claims: List[str]
    truths: List[str]
    verdicts: List[FaithfulnessVerdict]
