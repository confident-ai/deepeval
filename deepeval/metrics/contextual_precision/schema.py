from typing import List
from pydantic import BaseModel
from deepeval.metrics.base_metric import YesNo


class ContextualPrecisionVerdict(BaseModel):
    verdict: YesNo
    reason: str


class Verdicts(BaseModel):
    verdicts: List[ContextualPrecisionVerdict]


class ContextualPrecisionScoreReason(BaseModel):
    reason: str
