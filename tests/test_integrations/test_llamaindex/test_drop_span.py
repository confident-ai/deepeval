"""Regression tests for ``LLamaIndexHandler.prepare_to_drop_span``.

``llama_index_instrumentation`` calls ``span_drop(err=...)`` only from its
``except BaseException`` branches, so a dropped span is always a failed span.
Before the fix the handler ignored ``err``, stamped the span
``TraceSpanStatus.SUCCESS`` and never removed it from
``trace_manager.active_spans``.
"""

from __future__ import annotations

import uuid
from time import perf_counter
from unittest.mock import MagicMock

import pytest

import llama_index.core.instrumentation as instrument
from deepeval.integrations.llama_index.handler import LLamaIndexHandler
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import AgentSpan, TraceSpanStatus

dispatcher = instrument.get_dispatcher()


def _new_trace_with_root_span():
    trace = trace_manager.start_new_trace()
    span = AgentSpan(
        uuid=str(uuid.uuid4()),
        trace_uuid=trace.uuid,
        parent_uuid=None,
        start_time=perf_counter(),
        status=TraceSpanStatus.IN_PROGRESS,
        name="Workflow.run",
    )
    trace_manager.add_span(span)
    trace_manager.add_span_to_trace(span)
    return trace, span


def _drop_span(handler, span, err):
    return handler.prepare_to_drop_span(
        id_=span.uuid,
        bound_args=MagicMock(),
        instance=None,
        err=err,
    )


def test_dropped_span_records_error():
    """A span dropped with an exception is marked ERRORED and keeps the message."""
    handler = LLamaIndexHandler()
    _, span = _new_trace_with_root_span()

    returned = _drop_span(handler, span, ValueError("workflow step raised"))

    assert returned is span
    assert span.status == TraceSpanStatus.ERRORED
    assert span.error == "workflow step raised"


def test_dropped_span_is_removed_from_active_spans():
    """Dropping a span must not leak it into trace_manager.active_spans."""
    handler = LLamaIndexHandler()
    _, span = _new_trace_with_root_span()

    _drop_span(handler, span, ValueError("workflow step raised"))

    assert trace_manager.get_span_by_uuid(span.uuid) is None


def test_dropped_root_span_marks_trace_errored():
    """A failed root span must not leave its trace reported as successful."""
    handler = LLamaIndexHandler()
    trace, span = _new_trace_with_root_span()

    _drop_span(handler, span, ValueError("workflow step raised"))

    assert trace.status == TraceSpanStatus.ERRORED


def test_dropped_span_without_error_stays_successful():
    """A drop with no exception remains a successful span."""
    handler = LLamaIndexHandler()
    trace, span = _new_trace_with_root_span()

    _drop_span(handler, span, None)

    assert span.status == TraceSpanStatus.SUCCESS
    assert span.error is None
    assert trace.status == TraceSpanStatus.SUCCESS


@dispatcher.span
def _failing_operation():
    raise ValueError("downstream provider failed")


def test_instrumented_failure_ends_errored_trace(monkeypatch):
    """An instrumented call that raises must surface as an errored trace."""
    posted = []
    monkeypatch.setattr(trace_manager, "post_trace", posted.append)

    with pytest.raises(ValueError, match="downstream provider failed"):
        _failing_operation()

    assert len(posted) == 1
    trace = posted[0]
    assert trace.status == TraceSpanStatus.ERRORED
    assert trace.root_spans[0].status == TraceSpanStatus.ERRORED
    assert trace.root_spans[0].error == "downstream provider failed"
    assert trace_manager.active_spans == {}
