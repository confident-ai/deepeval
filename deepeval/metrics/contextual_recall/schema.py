from typing import List
from pydantic import BaseModel


class ContextualRecallVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[ContextualRecallVerdict]


class ContextualRecallScoreReason(BaseModel):
    reason: str
