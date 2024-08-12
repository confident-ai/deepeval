from pydantic import BaseModel, Field
from typing import List


class UserIntentions(BaseModel):
    intentions: List[str]


class ConversationCompletenessVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Reason(BaseModel):
    reason: str
