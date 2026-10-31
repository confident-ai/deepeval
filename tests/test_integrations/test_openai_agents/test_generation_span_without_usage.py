"""
Regression test: a generation span that finishes without usage data.

``GenerationSpanData.usage`` is ``Optional`` and is only assigned after a
successful model response, so a model call whose provider request raised
(bad API key, 429, connection error) finishes the span with ``usage=None``.
The agents SDK swallows trace-processor exceptions, so a crash in
``on_span_end`` closes nothing: the ``LlmSpan`` stays in
``current_span_context``, the trace-processor observer stays registered, and
every later span in the process parents onto the stale span.

No API key or network access is needed -- the failed provider call is
simulated by leaving ``span_data.usage`` at its default ``None``.
"""

import pytest

from agents.tracing import generation_span, trace

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


def test_generation_span_without_usage_is_finalized():
    with trace("generation_span_without_usage"):
        with generation_span(
            model="gpt-4o-mini", model_config={"temperature": 0.0}
        ) as span:
            span.span_data.input = [{"role": "user", "content": "hi"}]
            llm_span = current_span_context.get()

    assert llm_span.status == TraceSpanStatus.SUCCESS
    assert llm_span.end_time is not None
    assert llm_span.model == "gpt-4o-mini"
    assert llm_span.input_token_count is None
    assert llm_span.output_token_count is None
    assert llm_span.metadata["invocation_params"]["model_config"] == {
        "temperature": 0.0
    }


def test_generation_span_without_usage_leaves_nothing_open():
    with trace("generation_span_without_usage_open"):
        with generation_span(model="gpt-4o-mini") as span:
            span.span_data.input = [{"role": "user", "content": "hi"}]

    assert current_span_context.get() is None
    assert trace_manager.active_spans == {}
