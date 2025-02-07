from typing import Literal
from pydantic import BaseModel


class Reason(BaseModel):
    reason: str


class BinaryJudgementVerdict(BaseModel):
    verdict: Literal[True, False]
    reason: str


class NonBinaryJudgementVerdict(BaseModel):
    verdict: str
    reason: str
