from typing import List, Optional
from pydantic import BaseModel, Field
from deepeval.metrics.base_metric import YesNoBorderline


class Statements(BaseModel):
    statements: List[str]


class AnswerRelevancyVerdict(BaseModel):
    verdict: YesNoBorderline
    reason: Optional[str] = Field(default=None)


class Verdicts(BaseModel):
    verdicts: List[AnswerRelevancyVerdict]


class AnswerRelevancyScoreReason(BaseModel):
    reason: str
