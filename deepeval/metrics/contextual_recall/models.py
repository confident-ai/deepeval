from typing import List
from pydantic import BaseModel, Field


class ContextualRecallVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[ContextualRecallVerdict]


class Reason(BaseModel):
    reason: str
