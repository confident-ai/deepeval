from pydantic import BaseModel, Field
from typing import List


class Opinions(BaseModel):
    opinions: List[str]


# ToxicMetric uses similar rubric to decoding trust: https://arxiv.org/abs/2306.11698
class ToxicityVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[ToxicityVerdict]


class Reason(BaseModel):
    reason: str
