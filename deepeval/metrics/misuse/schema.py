from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Misuses(BaseModel):
    misuses: List[str]


class MisuseVerdict(BaseModel):
    verdict: Literal["yes", "no"]
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[MisuseVerdict]


class MisuseScoreReason(BaseModel):
    reason: str
