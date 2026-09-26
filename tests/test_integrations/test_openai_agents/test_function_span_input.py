"""Tests for span-property extraction from OpenAI Agents SDK span data.

``update_span_properties_from_function_span_data`` fed
``FunctionSpanData.input`` straight into ``json.loads(...)``. The SDK
types that field ``str | None`` (agents/tracing/span_data.py) and itself
emits function spans with ``input=None`` (custom tools / apply_patch via
``with_tool_function_span``, which never sets input) and ``input=""``
(a computer call with no actions — ``_serialize_trace_payload(None)``
returns ``""``). Both raised inside ``Observer.__exit__``, i.e. after
the span was added to ``trace_manager`` but before ``remove_span``
ran, so the ToolSpan leaked in ``active_spans`` under its placeholder
"NA" name. The SDK's ``DefaultTraceProvider`` swallows and logs
processor exceptions, so the corruption was silent.

The observer-level tests below drive the same ``Observer`` wiring
``DeepEvalTracingProcessor`` uses, with hand-built SDK spans — no API
key or network call needed.
"""

import pytest
from agents import gen_trace_id
from agents.tracing.create import function_span, trace
from agents.tracing.span_data import FunctionSpanData

from deepeval.openai_agents.extractors import (
    update_span_properties,
    update_span_properties_from_function_span_data,
)
from deepeval.tracing.context import (
    current_span_context,
    current_trace_context,
)
from deepeval.tracing.tracing import Observer, trace_manager
from deepeval.tracing.types import ToolSpan, TraceSpanStatus


def _reset_tracing_state():
    # a crashed span-end leaves spans/traces behind; reset so cases are
    # independent of each other
    for span_uuid in list(trace_manager.active_spans):
        trace_manager.remove_span(span_uuid)
    for trace_uuid in list(trace_manager.active_traces):
        trace_manager.end_trace(trace_uuid)
    current_span_context.set(None)
    current_trace_context.set(None)


@pytest.fixture(autouse=True)
def _clean_tracing_state():
    _reset_tracing_state()
    yield
    _reset_tracing_state()


def _end_function_span(span_input):
    """Run one SDK function span to span-end the way
    ``DeepEvalTracingProcessor`` does and return the ToolSpan it
    created.

    The SDK span is never started through the SDK itself, so processors
    registered with ``add_trace_processor`` (e.g. the session-wide one
    this directory's conftest installs) are not notified — only the
    ``Observer`` wiring under test runs.
    """
    sdk_trace = trace("test_trace", trace_id=gen_trace_id())
    sdk_span = function_span(
        "my_tool", input=span_input, output="tool ran", parent=sdk_trace
    )
    # same wiring as DeepEvalTracingProcessor.on_span_start
    observer = Observer(span_type="tool", func_name="NA")
    observer.update_span_properties = lambda span: update_span_properties(
        span, sdk_span.span_data
    )
    observer.__enter__()
    tool_span = trace_manager.active_spans[observer.uuid]
    # same call DeepEvalTracingProcessor.on_span_end makes through
    # Observer.__exit__
    observer.__exit__(None, None, None)
    return tool_span


class TestFunctionSpanInputThroughObserver:
    def test_input_none_does_not_raise(self):
        """``input=None`` is what the SDK emits for custom tools and
        apply_patch (``with_tool_function_span`` never sets input)."""
        tool_span = _end_function_span(None)

        assert tool_span.name == "Function tool: my_tool"
        assert tool_span.input == {"input": None}
        assert tool_span.output == "tool ran"
        assert tool_span.uuid not in trace_manager.active_spans

    def test_input_empty_string_does_not_raise(self):
        """``input=""`` is what the SDK emits for a computer call with
        no actions (``_serialize_trace_payload(None)``)."""
        tool_span = _end_function_span("")

        assert tool_span.name == "Function tool: my_tool"
        assert tool_span.input == {"input": ""}
        assert tool_span.output == "tool ran"
        assert tool_span.uuid not in trace_manager.active_spans

    def test_input_json_object_still_parsed(self):
        tool_span = _end_function_span('{"city": "SF"}')

        assert tool_span.name == "Function tool: my_tool"
        assert tool_span.input == {"city": "SF"}
        assert tool_span.uuid not in trace_manager.active_spans


class TestFunctionSpanInputFallback:
    """The raw input string must be kept whenever parsing it yields
    nothing — either a falsy JSON value or no JSON at all."""

    @staticmethod
    def _extract_input(span_input):
        span = ToolSpan(
            uuid="test-uuid",
            trace_uuid="test-trace-uuid",
            start_time=0.0,
            status=TraceSpanStatus.SUCCESS,
            name="NA",
        )
        update_span_properties_from_function_span_data(
            span,
            FunctionSpanData(name="my_tool", input=span_input, output=None),
        )
        return span.input

    def test_json_null_falls_back_to_raw(self):
        assert self._extract_input("null") == {"input": "null"}

    def test_non_json_string_falls_back_to_raw(self):
        assert self._extract_input("not json") == {"input": "not json"}

    def test_json_object_parsed(self):
        assert self._extract_input('{"city": "SF"}') == {"city": "SF"}
