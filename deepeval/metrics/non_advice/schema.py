from typing import List
from pydantic import BaseModel


class NonAdviceVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[NonAdviceVerdict]


class Advices(BaseModel):
    advices: List[str]


class NonAdviceScoreReason(BaseModel):
    reason: str
