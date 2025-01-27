from typing import Literal
from pydantic import BaseModel


class BinaryJudgementVerdict(BaseModel):
    verdict: Literal["YES", "NO"]
    reason: str


class NonBinaryJudgementVerdict(BaseModel):
    verdict: str
    reason: str
