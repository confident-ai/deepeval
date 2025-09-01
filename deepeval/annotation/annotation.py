from typing import Optional

from deepeval.confident.api import Api, Endpoints, HttpMethods
from deepeval.annotation.api import APIAnnotation, AnnotationType


def send_annotation(
    rating: int,
    trace_uuid: Optional[str] = None,
    span_uuid: Optional[str] = None,
    thread_id: Optional[str] = None,
    expected_output: Optional[str] = None,
    expected_outcome: Optional[str] = None,
    explanation: Optional[str] = None,
    user_id: Optional[str] = None,
    type: Optional[AnnotationType] = AnnotationType.THUMBS_RATING,
) -> str:
    api_annotation = APIAnnotation(
        rating=rating,
        traceUuid=trace_uuid,
        spanUuid=span_uuid,
        threadId=thread_id,
        expectedOutput=expected_output,
        expectedOutcome=expected_outcome,
        explanation=explanation,
        type=type,
        userId=user_id,
    )
    api = Api()
    try:
        body = api_annotation.model_dump(by_alias=True, exclude_none=True)
    except AttributeError:
        # Pydantic version below 2.0
        body = api_annotation.dict(by_alias=True, exclude_none=True)

    api.send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.ANNOTATIONS_ENDPOINT,
        body=body,
    )


async def a_send_annotation(
    rating: int,
    trace_uuid: Optional[str] = None,
    span_uuid: Optional[str] = None,
    thread_id: Optional[str] = None,
    expected_output: Optional[str] = None,
    expected_outcome: Optional[str] = None,
    explanation: Optional[str] = None,
    type: Optional[AnnotationType] = AnnotationType.THUMBS_RATING,
    user_id: Optional[str] = None,
) -> str:
    api_annotation = APIAnnotation(
        rating=rating,
        traceUuid=trace_uuid,
        spanUuid=span_uuid,
        threadId=thread_id,
        expectedOutput=expected_output,
        expectedOutcome=expected_outcome,
        explanation=explanation,
        type=type,
        userId=user_id,
    )
    api = Api()
    try:
        body = api_annotation.model_dump(by_alias=True, exclude_none=True)
    except AttributeError:
        # Pydantic version below 2.0
        body = api_annotation.dict(by_alias=True, exclude_none=True)

    await api.a_send_request(
        method=HttpMethods.POST,
        endpoint=Endpoints.ANNOTATIONS_ENDPOINT,
        body=body,
    )
