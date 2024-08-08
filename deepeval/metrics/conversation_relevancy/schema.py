from pydantic import BaseModel, Field


class ConversationRelevancyVerdict(BaseModel):
    verdict: str
    reason: str = Field(default=None)


class Reason(BaseModel):
    reason: str
