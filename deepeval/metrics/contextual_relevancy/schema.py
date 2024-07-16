from typing import List
from pydantic import BaseModel, Field


class ContextualRelevancyVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[ContextualRelevancyVerdict]


class Reason(BaseModel):
    reason: str
