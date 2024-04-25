from typing import Optional, List, Dict
from deepeval.api import Api, Endpoints
from deepeval.event.api import APIEvent, EventHttpResponse


def track(
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
    run_async: Optional[bool] = True,
) -> str:
    try:
        if additional_data and not all(
            isinstance(value, str) for value in additional_data.values()
        ):
            raise ValueError(
                "All values in the 'additional_data' must of type string."
            )

        if hyperparameters and not all(
            isinstance(value, str) for value in hyperparameters.values()
        ):
            raise ValueError(
                "All values in the 'hyperparameters' must of type string."
            )

        hyperparameters["model"] = model

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
        api = Api()
        try:
            body = api_event.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Pydantic version below 2.0
            body = api_event.dict(by_alias=True, exclude_none=True)

        result = api.post_request(
            endpoint=Endpoints.EVENT_ENDPOINT.value,
            body=body,
        )
        response = EventHttpResponse(eventId=result["eventId"])
        return response.eventId
    except Exception as e:
        if fail_silently:
            return

        if raise_expection:
            raise (e)
        else:
            print(str(e))
