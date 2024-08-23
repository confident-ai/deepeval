from typing import List, Optional
from pydantic import BaseModel, Field


class ContextualPrecisionVerdict(BaseModel):
    verdict: str
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[ContextualPrecisionVerdict]


class Reason(BaseModel):
    reason: str
