from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union
from enum import Enum

from deepeval.prompt import PromptApi


class Link(BaseModel):
    value: str


class CustomPropertyType(Enum):
    JSON = "JSON"
    LINK = "LINK"
    TEXT = "TEXT"


class CustomProperty(BaseModel):
    value: Union[str, Dict]
    type: CustomPropertyType

    class Config:
        use_enum_values = True


class APIEvent(BaseModel):
    name: str = Field(alias="name")
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
    custom_properties: Optional[
        Dict[str, Union[CustomProperty, List[CustomProperty]]]
    ] = Field(None, alias="customProperties")
    trace_stack: Optional[Dict] = Field(None, alias="traceStack")
    trace_provider: Optional[str] = Field(None, alias="traceProvider")
    hyperparameters: Optional[Dict[str, Union[str, PromptApi]]] = Field(None)

    class Config:
        use_enum_values = True


class APIFeedback(BaseModel):
    event_id: str = Field(alias="eventId")
    rating: Optional[int]
    expected_response: Optional[str] = Field(alias="expectedResponse")
    explanation: Optional[str] = Field(None)


class EventHttpResponse(BaseModel):
    eventId: str
