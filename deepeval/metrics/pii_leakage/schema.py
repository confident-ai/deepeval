from typing import List
from pydantic import BaseModel
from deepeval.metrics.base_metric import YesNo


class PIILeakageVerdict(BaseModel):
    verdict: YesNo
    reason: str


class Verdicts(BaseModel):
    verdicts: List[PIILeakageVerdict]


class ExtractedPII(BaseModel):
    extracted_pii: List[str]


class PIILeakageScoreReason(BaseModel):
    reason: str
