from typing import List
from pydantic import BaseModel, Field

class FaithfulnessVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)

class Verdicts(BaseModel):
    verdicts: List[FaithfulnessVerdict]

class Truth(BaseModel):
    truths: List[str]

class Claim(BaseModel):
    claims: List[str]

class Reason(BaseModel):
    reason: str