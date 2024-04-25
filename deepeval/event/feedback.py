from typing import Optional

from deepeval.api import Api, Endpoints
from deepeval.event.api import APIFeedback


def send_feedback(
    event_id: str,
    provider: str,
    rating: Optional[int] = None,
    expected_response: Optional[str] = None,
    explanation: Optional[str] = None,
    fail_silently: Optional[bool] = False,
    raise_expection: Optional[bool] = True,
) -> str:
    try:
        if provider != "user" and provider != "reviewer":
            raise ValueError("'provider' must be either 'user' or 'reviewer'.")

        if rating is None and expected_response is None and explanation is None:
            raise ValueError(
                "'rating', 'expected_response', and 'explanation' cannot all be None."
            )

        if rating < 0 or rating > 10:
            raise ValueError("'rating' must be between 0 and 10, inclusive.")

        api_event = APIFeedback(
            eventId=event_id,
            provider=provider,
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

        result = api.post_request(
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
