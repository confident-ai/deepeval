from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class PoisonSignals(BaseModel):
    poison_signals: List[str]


class PoisonVerdict(BaseModel):
    verdict: Literal["yes", "no"]
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[PoisonVerdict]


class PoisonScoreReason(BaseModel):
    reason: str
