from typing import List, Optional
from pydantic import BaseModel, Field


class HallucinationVerdict(BaseModel):
    verdict: str
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[HallucinationVerdict]


class Reason(BaseModel):
    reason: str
