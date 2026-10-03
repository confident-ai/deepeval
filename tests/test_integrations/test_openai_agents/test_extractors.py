"""Unit tests for the OpenAI Agents span-data extractors.

The OpenAI Agents SDK only populates ``ResponseSpanData.response`` and
``input`` when sensitive data is included in the trace (the default).
With ``RunConfig(trace_include_sensitive_data=False)`` it records only
``_response_id``, so every ``ResponseSpanData`` reaches the trace
processor with ``response=None`` and the extractors must tolerate it.
"""

from agents.tracing.span_data import ResponseSpanData

from deepeval.openai_agents.extractors import (
    update_trace_properties_from_span_data,
)
from deepeval.tracing.types import Trace, TraceSpanStatus


def test_update_trace_properties_without_response():
    trace = Trace(
        uuid="test-trace",
        status=TraceSpanStatus.IN_PROGRESS,
        root_spans=[],
        start_time=0.0,
    )
    span_data = ResponseSpanData(response=None, input=None)

    update_trace_properties_from_span_data(trace, span_data)

    assert trace.input is None
    assert trace.output is None
