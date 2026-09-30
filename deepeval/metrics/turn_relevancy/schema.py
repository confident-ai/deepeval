from typing import Optional

from pydantic import BaseModel, Field
from deepeval.metrics.base_metric import YesNo


class TurnRelevancyVerdict(BaseModel):
    verdict: YesNo
    reason: Optional[str] = Field(default=None)


class TurnRelevancyScoreReason(BaseModel):
    reason: str
