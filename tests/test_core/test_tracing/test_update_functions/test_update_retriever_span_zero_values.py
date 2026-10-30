from deepeval.tracing.context import (
    current_span_context,
    update_retriever_span,
)
from deepeval.tracing.types import RetrieverSpan, TraceSpanStatus


def _retriever_span() -> RetrieverSpan:
    return RetrieverSpan(
        uuid="span-uuid",
        trace_uuid="trace-uuid",
        parent_uuid=None,
        start_time=0.0,
        name="retriever",
        status=TraceSpanStatus.SUCCESS,
    )


class TestUpdateRetrieverSpanZeroValues:

    def test_zero_top_k_and_chunk_size_are_recorded(self):
        """``None`` is this signature's "not provided" marker, so a reported
        ``top_k=0`` has to survive instead of being treated as omitted."""
        span = _retriever_span()
        token = current_span_context.set(span)
        try:
            update_retriever_span(top_k=0, chunk_size=0)
        finally:
            current_span_context.reset(token)

        assert span.top_k == 0
        assert span.chunk_size == 0

    def test_omitted_fields_are_left_untouched(self):
        span = _retriever_span()
        span.top_k = 5
        span.chunk_size = 512
        token = current_span_context.set(span)
        try:
            update_retriever_span(embedder="text-embedding-3-small")
        finally:
            current_span_context.reset(token)

        assert span.embedder == "text-embedding-3-small"
        assert span.top_k == 5
        assert span.chunk_size == 512
