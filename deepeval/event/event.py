from typing import Optional, List, Dict, Union, Any

from deepeval.api import Api, Endpoints
from deepeval.test_run.hyperparameters import process_hyperparameters
from deepeval.event.api import (
    APIEvent,
    EventHttpResponse,
    CustomPropertyType,
    CustomProperty,
    Link,
)


from deepeval.api import Api, Endpoints
from deepeval.event.api import (
    APIEvent,
    EventHttpResponse,
    CustomPropertyType,
    CustomProperty,
    Link,
)


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
    additional_data: Optional[
        Dict[str, Union[str, Link, List[Link], Dict]]
    ] = None,
    hyperparameters: Optional[Dict[str, str]] = {},
    fail_silently: Optional[bool] = False,
    raise_expection: Optional[bool] = True,
    run_async: Optional[bool] = True,
    trace_stack: Optional[Dict[str, Any]] = None,
    trace_provider: Optional[str] = None,
) -> str:
    try:
        custom_properties = None
        if additional_data:
            custom_properties = {}
            for key, value in additional_data.items():
                if isinstance(value, str):
                    custom_properties[key] = CustomProperty(
                        value=value, type=CustomPropertyType.TEXT
                    )
                elif isinstance(value, dict):
                    custom_properties[key] = CustomProperty(
                        value=value, type=CustomPropertyType.JSON
                    )
                elif isinstance(value, Link):
                    custom_properties[key] = CustomProperty(
                        value=value.value, type=CustomPropertyType.LINK
                    )
                elif isinstance(value, list):
                    if not all(isinstance(item, Link) for item in value):
                        raise ValueError(
                            "All values in 'additional_data' must be either of type 'string', 'Link', list of 'Link', or 'dict'."
                        )
                    custom_properties[key] = [
                        CustomProperty(
                            value=item.value, type=CustomPropertyType.LINK
                        )
                        for item in value
                    ]
                else:
                    raise ValueError(
                        "All values in 'additional_data' must be either of type 'string', 'Link', list of 'Link', or 'dict'."
                    )

        hyperparameters = process_hyperparameters(hyperparameters)
        hyperparameters["model"] = model

        api_event = APIEvent(
            traceProvider=trace_provider,
            name=event_name,
            input=input,
            response=response,
            retrievalContext=retrieval_context,
            completionTime=completion_time,
            tokenUsage=token_usage,
            tokenCost=token_cost,
            distinctId=distinct_id,
            conversationId=conversation_id,
            customProperties=custom_properties,
            hyperparameters=hyperparameters,
            traceStack=trace_stack,
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
