from typing import List
from pydantic import BaseModel
from deepeval.metrics.base_metric import YesNo


class NonAdviceVerdict(BaseModel):
    verdict: YesNo
    reason: str


class Verdicts(BaseModel):
    verdicts: List[NonAdviceVerdict]


class Advices(BaseModel):
    advices: List[str]


class NonAdviceScoreReason(BaseModel):
    reason: str
