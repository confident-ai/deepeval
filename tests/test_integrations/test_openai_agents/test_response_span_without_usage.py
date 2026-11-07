"""
Regression test: a response span that finishes without usage data.

``Response.usage`` is ``Optional`` and OpenAI-compatible gateways and failed
model calls both produce responses with ``usage=None``. The agents SDK itself
guards this case, so a ``None`` usage is a legitimate input, not a malformed
one. Before the fix, ``update_span_properties_from_response_span_data``
raised ``UnboundLocalError`` because ``input_tokens``/``output_tokens`` were
only bound inside ``if usage:`` but read unconditionally. The agents SDK
swallows trace-processor exceptions, so a crash in ``on_span_end`` closes
nothing: the ``LlmSpan`` stays in ``current_span_context``, the
trace-processor observer stays registered, and every later span in the
process parents onto the stale span.

No API key or network access is needed -- the missing usage data is
simulated by building a ``Response`` with ``usage=None``.
"""

import pytest
from agents.tracing import response_span, trace
from openai.types.responses import Response

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


def test_response_span_without_usage_is_finalized():
    response = Response(
        id="resp_1",
        created_at=0,
        model="gpt-4o-mini",
        object="response",
        output=[],
        status="completed",
        usage=None,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    with trace("response_span_without_usage"), response_span(
        response=response
    ) as span:
        span.span_data.input = "hello"
        llm_span = current_span_context.get()

    assert llm_span.status == TraceSpanStatus.SUCCESS
    assert llm_span.end_time is not None
    assert llm_span.model == "gpt-4o-mini"
    assert llm_span.input_token_count is None
    assert llm_span.output_token_count is None
