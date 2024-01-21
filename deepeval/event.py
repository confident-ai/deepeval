from typing import Optional, List, Dict
from deepeval.api import Api, Endpoints
import threading
from pydantic import BaseModel, Field


class APIEvent(BaseModel):
    name: str = Field(..., alias="name")
    model: str
    input: str
    output: str
    retrieval_context: Optional[List[str]] = Field(
        None, alias="retrievalContext"
    )
    completion_time: Optional[float] = Field(None, alias="completionTime")
    token_usage: Optional[float] = Field(None, alias="tokenUsage")
    token_cost: Optional[float] = Field(None, alias="tokenCost")
    distinct_id: Optional[str] = Field(None, alias="distinctId")
    conversation_id: Optional[str] = Field(None, alias="conversationId")
    additional_data: Optional[Dict] = Field(None, alias="additionalData")


def track(
    event_name: str,
    model: str,
    input: str,
    output: str,
    retrieval_context: Optional[List[str]] = None,
    completion_time: Optional[float] = None,
    token_usage: Optional[float] = None,
    token_cost: Optional[float] = None,
    distinct_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    additional_data: Optional[Dict] = None,
    fail_silently: Optional[bool] = True,
    run_on_background_thread: Optional[bool] = True,
):
    def track_event(event: APIEvent, api: Api, fail_silently: bool):
        try:
            _ = api.post_request(
                endpoint=Endpoints.EVENT_ENDPOINT.value,
                body=event.dict(by_alias=True, exclude_none=True),
            )
        except Exception as e:
            if not fail_silently:
                raise e

    event = APIEvent(
        name=event_name,
        model=model,
        input=input,
        output=output,
        retrievalContext=retrieval_context,
        completionTime=completion_time,
        tokenUsage=token_usage,
        tokenCost=token_cost,
        distinctId=distinct_id,
        conversationId=conversation_id,
        additionalData=additional_data,
    )
    api = Api()
    if run_on_background_thread:
        thread = threading.Thread(
            target=track_event, args=(event, api, fail_silently), daemon=True
        )
        thread.start()
    else:
        track_event(event, api, fail_silently)
