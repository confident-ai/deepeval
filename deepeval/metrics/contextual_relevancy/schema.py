from typing import List, Optional
from pydantic import BaseModel, Field
# removed YesNo import


class ContextualRelevancyVerdict(BaseModel):
    statement: str
    verdict: Optional[str]
    reason: Optional[str] = Field(default=None)


class ContextualRelevancyVerdicts(BaseModel):
    verdicts: List[ContextualRelevancyVerdict]


class ContextualRelevancyScoreReason(BaseModel):
    reason: str
