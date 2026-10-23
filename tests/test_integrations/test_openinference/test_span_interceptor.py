"""Unit tests for OpenInference TOOL-span parity with Strands (#3103).

Mirrors the Strands/agentcore GenAI hardening: non-dict
``tool.parameters`` are wrapped to ``{"input": ...}`` (``null`` -> ``{}``)
instead of raising ``ValidationError`` (which ``on_end`` would swallow,
silently dropping the span's ``confident.*`` tool attrs), and the tool
output (``output.value``) is surfaced on ``tools_called[0].output`` the
same way Strands surfaces ``gen_ai.tool.call.result``.

These tests do NOT require any OpenInference instrumentor package —
they drive the interceptor with synthetic OTel spans built from
``MagicMock``.
"""

from __future__ import annotations

import json
from itertools import count
from unittest.mock import MagicMock

from deepeval.integrations.openinference.instrumentator import (
    OpenInferenceSpanInterceptor,
    _extract_tool_call_from_tool_span,
)
from deepeval.tracing.otel.attributes import ConfidentAttr


_span_id_counter = count(start=1)
_trace_id_counter = count(start=1)


def _make_mock_span(**attrs):
    """Mock OTel span shaped to match ``OpenInferenceSpanInterceptor``.

    Mirrors the OTel SDK invariant that ``Span.attributes`` is a view
    over the same underlying ``_attributes`` mapping — so writes via
    either ``set_attribute(...)`` or direct ``_attributes[k] = v``
    (used by ``_set_attr_post_end`` to bypass the ended-span guard)
    are observable via ``span.attributes.get(...)``.
    """
    span = MagicMock()
    backing: dict = dict(attrs)
    span._attributes = backing
    span.attributes = backing
    span.name = attrs.get("tool.name", "tool-span")
    span.events = []
    span.start_time = None  # forces _push_span_context to use perf_counter()
    span.parent = None  # None -> root span
    span.set_attribute.side_effect = lambda k, v: backing.__setitem__(k, v)
    span.get_span_context.return_value = MagicMock(
        trace_id=next(_trace_id_counter),
        span_id=next(_span_id_counter),
    )
    return span


def _make_settings(**kwargs):
    """Minimal mock ``OpenInferenceInstrumentationSettings``.

    Only fields ``OpenInferenceSpanInterceptor`` actually reads.
    ``spec=[]`` disallows auto-attrs so a typo on the interceptor
    side surfaces as AttributeError rather than a silent MagicMock.
    """
    settings = MagicMock(spec=[])
    settings.thread_id = kwargs.get("thread_id")
    settings.name = kwargs.get("name")
    settings.metadata = kwargs.get("metadata")
    settings.user_id = kwargs.get("user_id")
    settings.tags = kwargs.get("tags")
    settings.metric_collection = kwargs.get("metric_collection")
    settings.test_case_id = kwargs.get("test_case_id")
    settings.turn_id = kwargs.get("turn_id")
    settings.environment = kwargs.get("environment")
    settings.integration = kwargs.get("integration", "openinference")
    return settings


def _make_tool_span(tool_parameters=None, output_value=None):
    attrs = {"openinference.span.kind": "TOOL", "tool.name": "calc"}
    if tool_parameters is not None:
        attrs["tool.parameters"] = tool_parameters
    if output_value is not None:
        attrs["output.value"] = output_value
    return _make_mock_span(**attrs)


def _run_interceptor(span):
    interceptor = OpenInferenceSpanInterceptor(_make_settings())
    interceptor.on_start(span, None)
    interceptor.on_end(span)
    return span


def _tools_called(span):
    raw = span.attributes.get(ConfidentAttr.SPAN_TOOLS_CALLED)
    assert raw, "expected confident.span.tools_called to be recorded"
    return [json.loads(item) for item in raw]


class TestToolSpanParity:
    def test_list_args_wrapped_instead_of_raising(self):
        tc = _extract_tool_call_from_tool_span(_make_tool_span("[1, 2, 3]"))
        assert tc.input_parameters == {"input": [1, 2, 3]}

    def test_str_args_wrapped_instead_of_raising(self):
        tc = _extract_tool_call_from_tool_span(_make_tool_span('"hi"'))
        assert tc.input_parameters == {"input": "hi"}

    def test_null_args_become_empty_dict(self):
        tc = _extract_tool_call_from_tool_span(_make_tool_span("null"))
        assert tc.input_parameters == {}

    def test_malformed_args_become_empty_dict(self):
        tc = _extract_tool_call_from_tool_span(_make_tool_span("{bad"))
        assert tc.input_parameters == {}

    def test_tool_output_surfaced_on_tool_call(self):
        span = _run_interceptor(_make_tool_span('{"x": 1}', "42"))
        assert span.attributes.get(ConfidentAttr.SPAN_OUTPUT) == "42"
        assert _tools_called(span)[0]["output"] == "42"

    def test_non_dict_args_do_not_drop_span_attrs(self):
        """End-to-end guard: the pre-fix ValidationError aborted
        ``_serialize_framework_attrs`` mid-way, so the span lost its
        tool record while staying otherwise silent."""
        span = _run_interceptor(_make_tool_span("[1, 2, 3]", "42"))
        assert _tools_called(span)[0]["input_parameters"] == {
            "input": [1, 2, 3]
        }
