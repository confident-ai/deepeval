from time import perf_counter
from types import SimpleNamespace

from deepeval.openai_agents.extractors import (
    update_span_properties_from_response_span_data,
)
from deepeval.tracing.types import LlmSpan, TraceSpanStatus


def _make_span() -> LlmSpan:
    return LlmSpan(
        uuid="span-1",
        trace_uuid="trace-1",
        start_time=perf_counter(),
        status=TraceSpanStatus.IN_PROGRESS,
        children=[],
        name="NA",
        model="NA",
    )


def _make_response(usage):
    return SimpleNamespace(
        usage=usage,
        output=[],
        model="gpt-4o-mini",
        instructions="be brief",
        model_dump=lambda **kwargs: {"temperature": 0.0},
    )


def test_response_span_without_usage_sets_none_token_counts():
    # Response.usage is Optional in the OpenAI SDK; OpenAI-compatible
    # gateways and failed model calls return a response without usage.
    span = _make_span()
    span_data = SimpleNamespace(response=_make_response(usage=None), input="hi")

    update_span_properties_from_response_span_data(span, span_data)

    assert span.model == "gpt-4o-mini"
    assert span.input_token_count is None
    assert span.output_token_count is None
    assert span.metadata["cached_input_tokens"] is None
    assert span.metadata["ouptut_reasoning_tokens"] is None


def test_response_span_with_usage_sets_token_counts():
    usage = SimpleNamespace(
        input_tokens=11,
        output_tokens=7,
        input_tokens_details=SimpleNamespace(cached_tokens=3),
        output_tokens_details=SimpleNamespace(reasoning_tokens=2),
    )
    span = _make_span()
    span_data = SimpleNamespace(
        response=_make_response(usage=usage), input="hi"
    )

    update_span_properties_from_response_span_data(span, span_data)

    assert span.input_token_count == 11
    assert span.output_token_count == 7
    assert span.metadata["cached_input_tokens"] == 3
    assert span.metadata["ouptut_reasoning_tokens"] == 2
