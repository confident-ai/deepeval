from typing import Literal, Optional

from pydantic import BaseModel, Field


class TurnRelevancyVerdict(BaseModel):
    verdict: Literal["yes", "no"]
    reason: Optional[str] = Field(default=None)


class TurnRelevancyScoreReason(BaseModel):
    reason: str
