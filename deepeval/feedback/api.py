from pydantic import BaseModel, Field
from typing import Optional


class APIFeedback(BaseModel):
    rating: int
    trace_uuid: Optional[str] = Field(None, alias="traceUuid")
    span_uuid: Optional[str] = Field(None, alias="spanUuid")
    thread_supplied_id: Optional[str] = Field(None, alias="threadSuppliedId")
    expected_response: Optional[str] = Field(None, alias="expectedResponse")
    explanation: Optional[str] = Field(None)
