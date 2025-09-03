from pydantic import BaseModel, Field, model_validator
from typing import Optional
from enum import Enum


class AnnotationType(str, Enum):
    THUMBS_RATING = "THUMBS_RATING"
    FIVE_STAR_RATING = "FIVE_STAR_RATING"


class APIAnnotation(BaseModel):
    rating: int
    trace_uuid: Optional[str] = Field(None, alias="traceUuid")
    span_uuid: Optional[str] = Field(None, alias="spanUuid")
    thread_id: Optional[str] = Field(None, alias="threadId")
    expected_output: Optional[str] = Field(None, alias="expectedOutput")
    expected_outcome: Optional[str] = Field(None, alias="expectedOutcome")
    explanation: Optional[str] = Field(None)
    type: Optional[AnnotationType] = Field(None, alias="type")
    user_id: Optional[str] = Field(None, alias="userId")

    @model_validator(mode="before")
    def validate_input(cls, data):
        if (
            data.get("traceUuid")
            and data.get("spanUuid")
            and data.get("threadId")
        ):
            raise ValueError(
                "Only one of 'traceUuid', 'spanUuid', or 'threadId' should be provided."
            )
        if (
            not data.get("traceUuid")
            and not data.get("spanUuid")
            and not data.get("threadId")
        ):
            raise ValueError(
                "One of 'traceUuid', 'spanUuid', or 'threadId' must be provided."
            )
        if data.get("type") == AnnotationType.FIVE_STAR_RATING and (
            data.get("rating") < 1 or data.get("rating") > 5
        ):
            raise ValueError("Five star rating must be between 1 and 5.")
        if data.get("type") == AnnotationType.THUMBS_RATING and (
            data.get("rating") < 0 or data.get("rating") > 1
        ):
            raise ValueError("Thumbs rating must be either 0 or 1.")
        if data.get("threadId") and data.get("expectedOutput"):
            raise ValueError("Expected output cannot be provided for threads.")
        if not data.get("threadId") and data.get("expectedOutcome"):
            raise ValueError(
                "Expected outcome cannot be provided for traces or spans."
            )
        return data
