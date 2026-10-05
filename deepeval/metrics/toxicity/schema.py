from pydantic import BaseModel, Field
from deepeval.metrics.base_metric import YesNo
from typing import List, Optional


class Opinions(BaseModel):
    opinions: List[str]


# ToxicMetric uses similar rubric to decoding trust: https://arxiv.org/abs/2306.11698
class ToxicityVerdict(BaseModel):
    verdict: YesNo
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[ToxicityVerdict]


class ToxicityScoreReason(BaseModel):
    reason: str
