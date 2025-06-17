from typing import Optional
from pydantic import BaseModel

from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.feedback.api import APIFeedback


class Feedback(BaseModel):
    rating: int
    expected_output: Optional[str] = None
    explanation: Optional[str] = None


def collect_feedback(
    rating: int,
    trace_uuid: Optional[str] = None,
    span_uuid: Optional[str] = None,
    thread_id: Optional[str] = None,
    expected_output: Optional[str] = None,
    explanation: Optional[str] = None,
    fail_silently: Optional[bool] = False,
    raise_exception: Optional[bool] = True,
) -> str:
    try:
        id_count = sum(
            1 for id in [trace_uuid, span_uuid, thread_id] if id is not None
        )
        if id_count > 1:
            raise ValueError(
                "Only one of 'trace_uuid', 'span_uuid', or 'thread_id' can be provided"
            )
        if id_count == 0:
            raise ValueError(
                "One of 'trace_uuid', 'span_uuid', or 'thread_id' must be provided"
            )

        api_feedback = APIFeedback(
            rating=rating,
            traceUuid=trace_uuid,
            spanUuid=span_uuid,
            threadSuppliedId=thread_id,
            expectedResponse=expected_output,
            explanation=explanation,
        )
        api = Api()
        try:
            body = api_feedback.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Pydantic version below 2.0
            body = api_feedback.dict(by_alias=True, exclude_none=True)

        api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.FEEDBACK_ENDPOINT,
            body=body,
        )

        return
    except Exception as e:
        if fail_silently:
            return
        if raise_exception:
            raise (e)
        else:
            print(str(e))


async def a_collect_feedback(
    rating: int,
    trace_uuid: Optional[str] = None,
    span_uuid: Optional[str] = None,
    thread_id: Optional[str] = None,
    expected_output: Optional[str] = None,
    explanation: Optional[str] = None,
    fail_silently: Optional[bool] = False,
    raise_exception: Optional[bool] = True,
) -> str:
    try:
        id_count = sum(
            1 for id in [trace_uuid, span_uuid, thread_id] if id is not None
        )
        if id_count > 1:
            raise ValueError(
                "Only one of 'trace_uuid', 'span_uuid', or 'thread_id' can be provided"
            )
        if id_count == 0:
            raise ValueError(
                "One of 'trace_uuid', 'span_uuid', or 'thread_id' must be provided"
            )

        api_feedback = APIFeedback(
            rating=rating,
            traceUuid=trace_uuid,
            spanUuid=span_uuid,
            threadSuppliedId=thread_id,
            expectedResponse=expected_output,
            explanation=explanation,
        )
        api = Api()
        try:
            body = api_feedback.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            # Pydantic version below 2.0
            body = api_feedback.dict(by_alias=True, exclude_none=True)

        await api.a_send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.FEEDBACK_ENDPOINT,
            body=body,
        )

        return
    except Exception as e:
        if fail_silently:
            return
        if raise_exception:
            raise (e)
        else:
            print(str(e))
