from typing import List
from pydantic import BaseModel


class PIILeakageVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[PIILeakageVerdict]


class ExtractedPII(BaseModel):
    extracted_pii: List[str]


class PIILeakageScoreReason(BaseModel):
    reason: str
