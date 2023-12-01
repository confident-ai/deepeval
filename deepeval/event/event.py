from typing import Optional, List, Dict
from deepeval.api import Api, Endpoints
import asyncio
from .api import APIEvent

async def track_async(
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
):
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
    try:
        _ = api.post_request(
            endpoint=Endpoints.CREATE_EVENT_ENDPOINT.value,
            body=event.dict(by_alias=True, exclude_none=True),
        )
    except Exception as e:
        if not fail_silently:
            raise (e)


def track(*args, **kwargs):
    return asyncio.run(track_async(*args, **kwargs))