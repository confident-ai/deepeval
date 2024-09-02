from typing import List, Optional
from pydantic import BaseModel, Field


class FaithfulnessVerdict(BaseModel):
    verdict: str
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[FaithfulnessVerdict]


class Truths(BaseModel):
    truths: List[str]


class Claims(BaseModel):
    claims: List[str]


class Reason(BaseModel):
    reason: str
