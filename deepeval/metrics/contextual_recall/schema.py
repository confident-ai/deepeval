from typing import List, Optional
from pydantic import BaseModel, Field


class ContextualRecallVerdict(BaseModel):
    verdict: str
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[ContextualRecallVerdict]


class Reason(BaseModel):
    reason: str
