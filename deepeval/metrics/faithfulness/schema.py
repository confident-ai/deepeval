from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class FaithfulnessVerdict(BaseModel):
    verdict: Literal["yes", "no", "idk"]
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[FaithfulnessVerdict]


class Truths(BaseModel):
    truths: List[str]


class Claims(BaseModel):
    claims: List[str]


class FaithfulnessScoreReason(BaseModel):
    reason: str
