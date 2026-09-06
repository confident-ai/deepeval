from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class EscalationSignals(BaseModel):
    escalation_signals: List[str]


class EscalationVerdict(BaseModel):
    verdict: Literal["yes", "no"]
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[EscalationVerdict]


class EscalationScoreReason(BaseModel):
    reason: str
