from typing import List
from pydantic import BaseModel
from deepeval.metrics.base_metric import YesNo


class HallucinationVerdict(BaseModel):
    verdict: YesNo
    reason: str


class Verdicts(BaseModel):
    verdicts: List[HallucinationVerdict]


class HallucinationScoreReason(BaseModel):
    reason: str
