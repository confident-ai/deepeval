from typing import Optional

from pydantic import BaseModel, Field


class TurnRelevancyVerdict(BaseModel):
    verdict: str
    reason: Optional[str] = Field(default=None)


class TurnRelevancyScoreReason(BaseModel):
    reason: str
