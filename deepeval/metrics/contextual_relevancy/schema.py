from typing import List, Optional
from pydantic import BaseModel, Field
from deepeval.metrics.base_metric import YesNo


class ContextualRelevancyVerdict(BaseModel):
    statement: str
    verdict: YesNo
    reason: Optional[str] = Field(default=None)


class ContextualRelevancyVerdicts(BaseModel):
    verdicts: List[ContextualRelevancyVerdict]


class ContextualRelevancyScoreReason(BaseModel):
    reason: str
