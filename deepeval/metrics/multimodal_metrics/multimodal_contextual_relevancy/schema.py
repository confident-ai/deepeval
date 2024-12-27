from typing import List, Optional
from pydantic import BaseModel, Field


class ContextualRelevancyVerdict(BaseModel):
    statement: str
    verdict: str
    reason: Optional[str] = Field(default=None)


class ContextualRelevancyVerdicts(BaseModel):
    verdicts: List[ContextualRelevancyVerdict]


class Reason(BaseModel):
    reason: str
