from typing import List, Optional
from pydantic import BaseModel, Field


class HallucinationVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[HallucinationVerdict]


class Reason(BaseModel):
    reason: str
