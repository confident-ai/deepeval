from typing import List
from pydantic import BaseModel, Field


class ContextualPrecisionVerdict(BaseModel):
    verdict: str
    reason: str


class Verdicts(BaseModel):
    verdicts: List[ContextualPrecisionVerdict]


class Reason(BaseModel):
    reason: str
