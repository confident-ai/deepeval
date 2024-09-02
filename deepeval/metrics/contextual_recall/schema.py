from typing import List, Optional
from pydantic import BaseModel, Field


class ContextualRecallVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[ContextualRecallVerdict]


class Reason(BaseModel):
    reason: str
