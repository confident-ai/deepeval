from deepeval.tracing.context import current_span_context, update_llm_span
from deepeval.tracing.types import LlmSpan, TraceSpanStatus


def _llm_span() -> LlmSpan:
    return LlmSpan(
        uuid="span-uuid",
        trace_uuid="trace-uuid",
        parent_uuid=None,
        start_time=0.0,
        name="llm",
        status=TraceSpanStatus.SUCCESS,
        model="local/llama3",
    )


class TestUpdateLlmSpanZeroValues:

    def test_zero_rates_and_token_counts_are_recorded(self):
        """A free model reports a per-token rate of 0.0; that is a value, not
        an absent value, and must not be dropped."""
        span = _llm_span()
        token = current_span_context.set(span)
        try:
            update_llm_span(
                input_token_count=0,
                output_token_count=0,
                cost_per_input_token=0.0,
                cost_per_output_token=0.0,
            )
        finally:
            current_span_context.reset(token)

        assert span.input_token_count == 0
        assert span.output_token_count == 0
        assert span.cost_per_input_token == 0.0
        assert span.cost_per_output_token == 0.0

    def test_omitted_fields_are_left_untouched(self):
        span = _llm_span()
        span.input_token_count = 12
        span.cost_per_input_token = 0.5
        token = current_span_context.set(span)
        try:
            update_llm_span(output_token_count=7)
        finally:
            current_span_context.reset(token)

        assert span.input_token_count == 12
        assert span.cost_per_input_token == 0.5
        assert span.output_token_count == 7
