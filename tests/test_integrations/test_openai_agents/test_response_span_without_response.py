"""
Regression test: a response span that finishes without a Response object.

``ResponseSpanData.response`` is ``Optional`` -- a failed Responses API call
never produces a ``Response``, so the agents SDK ends the span with
``response=None``. Before the fix, ``update_trace_properties_from_span_data``
dereferenced ``span_data.response.instructions`` unconditionally, so
``on_span_end`` raised ``AttributeError``. The agents SDK swallows
trace-processor exceptions, so the crash closed nothing: the ``LlmSpan``
stayed open (``end_time`` unset) and every later span in the process
parented onto the stale span.

No API key or network access is needed -- the missing response is simulated
by entering a ``response_span`` without a response, exactly as a failed
model call does.
"""

import pytest
from agents.tracing import response_span, trace

from deepeval.tracing.context import (
    current_span_context,
    current_trace_context,
)
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import TraceSpanStatus


@pytest.fixture(autouse=True)
def _reset_tracing_state():
    current_span_context.set(None)
    current_trace_context.set(None)
    trace_manager.clear_traces()
    yield
    current_span_context.set(None)
    current_trace_context.set(None)
    trace_manager.clear_traces()


def test_response_span_without_response_is_finalized():
    with trace("response_span_without_response"), response_span() as span:
        span.span_data.input = "hello"
        llm_span = current_span_context.get()

    assert llm_span.status == TraceSpanStatus.SUCCESS
    assert llm_span.end_time is not None
    assert llm_span.model == "NA"
    assert llm_span.input_token_count is None
    assert llm_span.output_token_count is None
