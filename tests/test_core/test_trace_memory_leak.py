"""Tests for trace lifecycle cleanup in trace_manager.traces."""

import pytest

from deepeval.tracing import observe, trace_manager
from deepeval.tracing.context import current_span_context, current_trace_context


@observe(type="agent")
def simple_agent():
    return "done"


@observe(type="agent")
def agent_with_child():
    @observe(type="tool")
    def tool_call():
        return "tool result"

    tool_call()
    return "done"


@pytest.fixture(autouse=True)
def clean_state():
    trace_manager.clear_traces()
    current_span_context.set(None)
    current_trace_context.set(None)
    yield
    trace_manager.clear_traces()
    current_span_context.set(None)
    current_trace_context.set(None)


class TestTraceCleanup:
    """Completed traces are evicted from trace_manager.traces."""

    def test_single_trace_removed_after_completion(self):
        simple_agent()

        assert len(trace_manager.traces) == 0
        assert len(trace_manager.active_traces) == 0

    def test_nested_spans_trace_removed_after_completion(self):
        agent_with_child()

        assert len(trace_manager.traces) == 0
        assert len(trace_manager.active_traces) == 0
        assert len(trace_manager.active_spans) == 0

    def test_many_traces_do_not_accumulate(self):
        for _ in range(200):
            simple_agent()

        assert (
            len(trace_manager.traces) == 0
        ), f"Expected 0 retained traces, got {len(trace_manager.traces)}"

    def test_active_traces_cleaned_up(self):
        for _ in range(50):
            agent_with_child()

        assert len(trace_manager.active_traces) == 0
        assert len(trace_manager.active_spans) == 0


class TestTraceRetentionDuringEvaluation:
    """Traces remain in trace_manager.traces during evaluation mode."""

    @pytest.fixture(autouse=True)
    def _evaluation_mode(self):
        from deepeval.tracing.types import EvalMode, EvalSession

        trace_manager.eval_session = EvalSession(mode=EvalMode.EVALUATE)
        yield
        trace_manager.eval_session = EvalSession()

    def test_traces_retained_during_evaluation(self):
        simple_agent()

        assert len(trace_manager.traces) == 1
