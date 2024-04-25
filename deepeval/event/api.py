from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class APIEvent(BaseModel):
    name: str = Field(..., alias="name")
    input: str
    response: str
    retrieval_context: Optional[List[str]] = Field(
        None, alias="retrievalContext"
    )
    completion_time: Optional[float] = Field(None, alias="completionTime")
    token_usage: Optional[float] = Field(None, alias="tokenUsage")
    token_cost: Optional[float] = Field(None, alias="tokenCost")
    distinct_id: Optional[str] = Field(None, alias="distinctId")
    conversation_id: Optional[str] = Field(None, alias="conversationId")
    additional_data: Optional[Dict] = Field(None, alias="additionalData")
    hyperparameters: Optional[Dict] = Field(None)


class APIFeedback(BaseModel):
    provider: str
    event_id: str = Field(alias="eventId")
    rating: Optional[int]
    expected_response: Optional[str] = Field(alias="expectedResponse")
    explanation: Optional[str] = Field(None)


class EventHttpResponse(BaseModel):
    eventId: str
