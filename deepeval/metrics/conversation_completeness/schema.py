from typing import List, Optional

from pydantic import BaseModel, Field


class UserIntentions(BaseModel):
    intentions: List[str]


class ConversationCompletenessVerdict(BaseModel):
    verdict: str
    reason: Optional[str] = Field(default=None)


class Reason(BaseModel):
    reason: str
