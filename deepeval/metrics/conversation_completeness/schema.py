from pydantic import BaseModel, Field
from deepeval.metrics.base_metric import YesNo
from typing import List, Optional


class UserIntentions(BaseModel):
    intentions: List[str]


class ConversationCompletenessVerdict(BaseModel):
    verdict: YesNo
    reason: Optional[str] = Field(default=None)


class ConversationCompletenessScoreReason(BaseModel):
    reason: str
