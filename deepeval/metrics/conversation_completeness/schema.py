from pydantic import BaseModel, Field
from typing import List, Optional


class UserIntentions(BaseModel):
    intentions: List[str]


class ConversationCompletenessVerdict(BaseModel):
    verdict: str
    reason: Optional[str] = Field(default=None)


class Reason(BaseModel):
    reason: str
