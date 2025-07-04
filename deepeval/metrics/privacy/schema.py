from typing import List
from pydantic import BaseModel


class PrivacyVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[PrivacyVerdict]


class Reason(BaseModel):
    reason: str 