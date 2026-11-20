import time

import pytest

from deepeval.metrics.community import ToolOutcomeMetric
from deepeval.test_case import LLMTestCase
from deepeval.tracing.tracing import trace_manager
from deepeval.tracing.types import AgentSpan, ToolSpan, TraceSpanStatus


def _tool_span(name, error=None):
    span = {"type": "tool", "name": name, "input": {}, "children": []}
    if error is not None:
        span["error"] = error
    return span


def _test_case(*tool_spans):
    test_case = LLMTestCase(input="do the task", actual_output="done")
    test_case._trace_dict = {
        "type": "agent",
        "name": "agent",
        "children": list(tool_spans),
    }
    return test_case


class TestToolOutcomeMetric:
    """ToolOutcomeMetric is deterministic, so these run without an API key."""

    def test_all_calls_succeed_passes(self):
        metric = ToolOutcomeMetric()
        metric.measure(_test_case(_tool_span("search"), _tool_span("reply")))
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_errored_call_gives_partial_score(self):
        metric = ToolOutcomeMetric()
        metric.measure(
            _test_case(
                _tool_span("search"),
                _tool_span("payment_api", error="TimeoutError: 30s"),
            )
        )
        assert metric.score == 0.5
        assert metric.is_successful() is False
        assert "payment_api (1)" in metric.reason

    def test_partial_credit_with_threshold(self):
        metric = ToolOutcomeMetric(threshold=0.6)
        metric.measure(
            _test_case(
                _tool_span("a"),
                _tool_span("b"),
                _tool_span("c", error="boom"),
            )
        )
        assert round(metric.score, 2) == 0.67
        assert metric.is_successful() is True

    def test_recovered_failure_counts_by_default(self):
        metric = ToolOutcomeMetric()
        metric.measure(
            _test_case(
                _tool_span("search", error="503"),
                _tool_span("search"),
            )
        )
        assert metric.score == 0.5

    def test_ignore_recovered_failures(self):
        metric = ToolOutcomeMetric(ignore_recovered_failures=True)
        metric.measure(
            _test_case(
                _tool_span("search", error="503"),
                _tool_span("search"),
            )
        )
        assert metric.score == 1.0
        assert "recovered" in metric.reason

    def test_success_before_failure_does_not_recover(self):
        metric = ToolOutcomeMetric(ignore_recovered_failures=True)
        metric.measure(
            _test_case(
                _tool_span("search"),
                _tool_span("search", error="503"),
            )
        )
        assert metric.score == 0.5

    def test_success_of_other_tool_does_not_recover(self):
        metric = ToolOutcomeMetric(ignore_recovered_failures=True)
        metric.measure(
            _test_case(
                _tool_span("search", error="503"),
                _tool_span("reply"),
            )
        )
        assert metric.score == 0.5

    def test_nested_tool_spans_are_counted(self):
        nested = _tool_span("outer")
        nested["children"] = [_tool_span("inner", error="boom")]
        metric = ToolOutcomeMetric()
        metric.measure(_test_case(nested))
        assert metric.score == 0.5
        assert "inner (1)" in metric.reason

    def test_no_tool_calls_passes(self):
        metric = ToolOutcomeMetric()
        metric.measure(_test_case())
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_missing_trace_fails(self):
        metric = ToolOutcomeMetric()
        metric.measure(LLMTestCase(input="do the task", actual_output="done"))
        assert metric.score == 0.0
        assert metric.is_successful() is False
        assert "No trace data" in metric.reason

    def test_strict_mode_zeroes_partial_success(self):
        metric = ToolOutcomeMetric(strict_mode=True)
        metric.measure(
            _test_case(_tool_span("a"), _tool_span("b", error="boom"))
        )
        assert metric.score == 0
        assert metric.is_successful() is False

    def test_requires_trace(self):
        assert ToolOutcomeMetric().requires_trace is True

    def test_errors_from_real_trace_spans_are_detected(self):
        now = time.perf_counter()
        common = dict(trace_uuid="trace", start_time=now, end_time=now)
        root = AgentSpan(
            uuid="root", name="agent", status=TraceSpanStatus.SUCCESS, **common
        )
        root.children = [
            ToolSpan(
                uuid="ok",
                name="search",
                status=TraceSpanStatus.SUCCESS,
                parent_uuid="root",
                **common,
            ),
            ToolSpan(
                uuid="bad",
                name="payment_api",
                status=TraceSpanStatus.ERRORED,
                error="ConnectionError: refused",
                parent_uuid="root",
                **common,
            ),
        ]
        test_case = LLMTestCase(input="do the task", actual_output="done")
        test_case._trace_dict = trace_manager.create_nested_spans_dict(root)

        metric = ToolOutcomeMetric()
        metric.measure(test_case)
        assert metric.score == 0.5
        assert "payment_api (1)" in metric.reason

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = ToolOutcomeMetric()
        score = await metric.a_measure(
            _test_case(_tool_span("a"), _tool_span("b", error="boom"))
        )
        assert score == 0.5
        assert metric.is_successful() is False
