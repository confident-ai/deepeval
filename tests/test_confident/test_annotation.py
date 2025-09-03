from deepeval.annotation import send_annotation
from deepeval.annotation.api import AnnotationType
from deepeval.confident.api import ConfidentApiError
import pytest

VALID_TRACE_UUID = "b1c32524-a985-42e9-a9f0-fbcdf3ad0e55"
VALID_SPAN_UUID = "dffdee6c-eda2-459e-bab3-ec6793c6f5de"
VALID_THREAD_ID = "131324ljihfsadiuyip"

INVALID_TRACE_UUID = "123"
INVALID_SPAN_UUID = "123"
INVALID_THREAD_ID = "123"

TEST_USER_ID = "test_user_id"


class TestTraceAnnotation:
    def test_annotate_trace_with_thumbs_rating_invalid_uuid(self):
        with pytest.raises(ConfidentApiError):
            send_annotation(
                trace_uuid=INVALID_TRACE_UUID,
                expected_output="This is a test annotation",
                rating=1,
            )

    def test_annotate_trace_with_thumbs_rating_invalid_rating(self):
        with pytest.raises(ValueError):
            send_annotation(
                trace_uuid=VALID_TRACE_UUID,
                expected_output="This is a test annotation",
                type=AnnotationType.THUMBS_RATING,
                rating=3,
            )

    def test_annotate_trace_with_thumbs_rating_valid(self):
        send_annotation(
            trace_uuid=VALID_TRACE_UUID,
            expected_output="This is a test annotation",
            type=AnnotationType.THUMBS_RATING,
            rating=1,
        )

    def test_annotate_trace_with_five_star_rating_invalid_rating(self):
        with pytest.raises(ValueError):
            send_annotation(
                trace_uuid=VALID_TRACE_UUID,
                expected_output="This is a test annotation",
                type=AnnotationType.FIVE_STAR_RATING,
                rating=6,
            )

    def test_annotate_trace_with_five_star_rating_invalid_uuid(self):
        with pytest.raises(ValueError):
            send_annotation(
                trace_uuid=INVALID_TRACE_UUID,
                expected_output="This is a test annotation",
                type=AnnotationType.FIVE_STAR_RATING,
                rating=6,
            )

    def test_annotate_trace_with_five_star_rating_valid(self):
        send_annotation(
            trace_uuid=VALID_TRACE_UUID,
            expected_output="This is a test annotation",
            type=AnnotationType.FIVE_STAR_RATING,
            rating=5,
        )

    def test_annotate_trace_with_user_id(self):
        send_annotation(
            trace_uuid=VALID_TRACE_UUID,
            rating=1,
            user_id=TEST_USER_ID,
        )


class TestSpanAnnotation:
    def test_annotate_span_valid(self):
        send_annotation(
            span_uuid=VALID_SPAN_UUID,
            expected_output="This is a test annotation",
            rating=1,
        )

    def test_annotate_span_invalid_uuid(self):
        with pytest.raises(ConfidentApiError):
            send_annotation(
                span_uuid=INVALID_SPAN_UUID,
                expected_output="This is a test annotation",
                rating=1,
            )

    def test_annotate_span_with_thumbs_rating_invalid_uuid(self):
        with pytest.raises(ConfidentApiError):
            send_annotation(
                span_uuid=INVALID_SPAN_UUID,
                expected_output="This is a test annotation",
                rating=1,
            )

    def test_annotate_span_with_thumbs_rating_invalid_rating(self):
        with pytest.raises(ValueError):
            send_annotation(
                span_uuid=VALID_SPAN_UUID,
                expected_output="This is a test annotation",
                type=AnnotationType.THUMBS_RATING,
                rating=3,
            )

    def test_annotate_span_with_thumbs_rating_valid(self):
        send_annotation(
            span_uuid=VALID_SPAN_UUID,
            expected_output="This is a test annotation",
            type=AnnotationType.THUMBS_RATING,
            rating=1,
        )

    def test_annotate_span_with_five_star_rating_invalid_rating(self):
        with pytest.raises(ValueError):
            send_annotation(
                span_uuid=VALID_SPAN_UUID,
                expected_output="This is a test annotation",
                type=AnnotationType.FIVE_STAR_RATING,
                rating=6,
            )

    def test_annotate_span_with_five_star_rating_invalid_uuid(self):
        with pytest.raises(ValueError):
            send_annotation(
                span_uuid=INVALID_SPAN_UUID,
                expected_output="This is a test annotation",
                type=AnnotationType.FIVE_STAR_RATING,
                rating=6,
            )

    def test_annotate_span_with_five_star_rating_valid(self):
        send_annotation(
            span_uuid=VALID_SPAN_UUID,
            expected_output="This is a test annotation",
            type=AnnotationType.FIVE_STAR_RATING,
            rating=5,
        )

    def test_annotate_span_with_user_id(self):
        send_annotation(
            span_uuid=VALID_SPAN_UUID,
            rating=1,
            user_id=TEST_USER_ID,
        )


class TestThreadAnnotation:
    def test_annotate_thread_valid(self):
        send_annotation(
            thread_id=VALID_THREAD_ID,
            expected_outcome="This is a test annotation",
            rating=1,
        )

    def test_annotate_thread_invalid_id(self):
        with pytest.raises(ConfidentApiError):
            send_annotation(
                thread_id=INVALID_THREAD_ID,
                expected_outcome="This is a test annotation",
                rating=1,
            )

    def test_annotate_thread_with_thumbs_rating_invalid_id(self):
        with pytest.raises(ValueError):
            send_annotation(
                thread_id=INVALID_THREAD_ID,
                expected_outcome="This is a test annotation",
                type=AnnotationType.THUMBS_RATING,
                rating=3,
            )

    def test_annotate_thread_with_thumbs_rating_invalid_rating(self):
        with pytest.raises(ValueError):
            send_annotation(
                thread_id=VALID_THREAD_ID,
                expected_outcome="This is a test annotation",
                type=AnnotationType.THUMBS_RATING,
                rating=3,
            )

    def test_annotate_thread_with_thumbs_rating_valid(self):
        send_annotation(
            thread_id=VALID_THREAD_ID,
            expected_outcome="This is a test annotation",
            type=AnnotationType.THUMBS_RATING,
            rating=1,
        )

    def test_annotate_thread_with_five_star_rating_invalid_rating(self):
        with pytest.raises(ValueError):
            send_annotation(
                thread_id=VALID_THREAD_ID,
                expected_outcome="This is a test annotation",
                type=AnnotationType.FIVE_STAR_RATING,
                rating=6,
            )

    def test_annotate_thread_with_five_star_rating_invalid_id(self):
        with pytest.raises(ValueError):
            send_annotation(
                thread_id=INVALID_THREAD_ID,
                expected_outcome="This is a test annotation",
                type=AnnotationType.FIVE_STAR_RATING,
                rating=6,
            )

    def test_annotate_thread_with_five_star_rating_valid(self):
        send_annotation(
            thread_id=VALID_THREAD_ID,
            expected_outcome="This is a test annotation",
            type=AnnotationType.FIVE_STAR_RATING,
            rating=5,
        )

    def test_annotate_thread_with_user_id(self):
        send_annotation(
            thread_id=VALID_THREAD_ID,
            rating=1,
            user_id=TEST_USER_ID,
        )
