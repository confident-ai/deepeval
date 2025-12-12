from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class FaithfulnessVerdict(BaseModel):
    reason: Optional[str] = Field(default=None)
    verdict: Literal["yes", "no", "idk"]


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
