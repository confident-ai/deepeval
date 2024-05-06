from typing import Optional, List, Dict
from deepeval.api import Api, Endpoints
from deepeval.event.api import APIEvent, EventHttpResponse


async def track(
    event_name: str,
    model: str,
    input: str,
    response: str,
    retrieval_context: Optional[List[str]] = None,
    completion_time: Optional[float] = None,
    token_usage: Optional[float] = None,
    token_cost: Optional[float] = None,
    distinct_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    additional_data: Optional[Dict[str, str]] = None,
    hyperparameters: Optional[Dict[str, str]] = {},
    fail_silently: Optional[bool] = False,
    raise_expection: Optional[bool] = True,
    run_async: Optional[bool] = True
) -> Optional[str]:
    try:
        # Check data types for additional_data and hyperparameters
        if additional_data and not all(
            isinstance(value, str) for value in additional_data.values()
        ):
            raise ValueError(
                "All values in 'additional_data' must be of type string."
            )

        if hyperparameters and not all(
            isinstance(value, str) for value in hyperparameters.values()
        ):
            raise ValueError(
                "All values in 'hyperparameters' must be of type string."
            )

        hyperparameters["model"] = model

        # Prepare the event data
        api_event = APIEvent(
            name=event_name,
            input=input,
            response=response,
            retrievalContext=retrieval_context,
            completionTime=completion_time,
            tokenUsage=token_usage,
            tokenCost=token_cost,
            distinctId=distinct_id,
            conversationId=conversation_id,
            additionalData=additional_data,
            hyperparameters=hyperparameters,
        )
        
        # Try to serialize the event using Pydantic's model_dump (or dict for older versions)
        try:
            body = api_event.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Fallback for older Pydantic versions
            body = api_event.dict(by_alias=True, exclude_none=True)

        # Create an instance of the Api class to use its post_request or post_request_async
        api = Api()

        if run_async:
            # Asynchronous request
            result = await api.post_request_async(Endpoints.EVENT_ENDPOINT.value, body=body)
        else:
            # Synchronous request
            result = api.post_request(Endpoints.EVENT_ENDPOINT.value, body=body)

        # Parse the response and return the event ID
        if result is not None:
            event_id = result.get("eventId")
            return event_id
        else:
            return None

    except Exception as e:
        if fail_silently:
            return None

        if raise_expection:
            raise e
        else:
            print(str(e))
            return None