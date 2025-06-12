from typing import Optional
from deepeval.feedback.feedback import Feedback
from deepeval.feedback.api import APIFeedback


def convert_feedback_to_api_feedback(
    feedback: Optional[Feedback] = None,
    trace_uuid: Optional[str] = None,
    span_uuid: Optional[str] = None,
) -> Optional[APIFeedback]:
    if feedback is None:
        return None

    if trace_uuid is not None:
        return APIFeedback(
            traceUuid=trace_uuid,
            rating=feedback.rating,
            expectedResponse=feedback.expected_output,
            explanation=feedback.explanation,
        )
    if span_uuid is not None:
        return APIFeedback(
            spanUuid=span_uuid,
            rating=feedback.rating,
            expectedResponse=feedback.expected_output,
            explanation=feedback.explanation,
        )
    else:
        return None
