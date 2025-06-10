from pydantic import BaseModel, Field
from typing import Optional


class APIFeedback(BaseModel):
    event_id: str = Field(alias="eventId")
    rating: Optional[int]
    expected_response: Optional[str] = Field(alias="expectedResponse")
    explanation: Optional[str] = Field(None)
