from typing import Optional

from deepeval.api import Api, Endpoints
from deepeval.event.api import APIFeedback


def send_feedback(
    event_id: str,
    rating: int,
    expected_response: Optional[str] = None,
    explanation: Optional[str] = None,
    fail_silently: Optional[bool] = False,
    raise_expection: Optional[bool] = True,
) -> str:
    try:
        api_event = APIFeedback(
            eventId=event_id,
            rating=rating,
            expectedResponse=expected_response,
            explanation=explanation,
        )
        api = Api()
        try:
            body = api_event.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Pydantic version below 2.0
            body = api_event.dict(by_alias=True, exclude_none=True)

        api.post_request(
            endpoint=Endpoints.FEEDBACK_ENDPOINT.value,
            body=body,
        )

        return
    except Exception as e:
        if fail_silently:
            return

        if raise_expection:
            raise (e)
        else:
            print(str(e))
