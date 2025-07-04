from typing import List
from pydantic import BaseModel


class NonAdviceVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[NonAdviceVerdict]


class Reason(BaseModel):
    reason: str 