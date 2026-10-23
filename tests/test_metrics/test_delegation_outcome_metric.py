import pytest

from deepeval.metrics.community import DelegationOutcomeMetric
from deepeval.test_case import LLMTestCase


def _agent_span(name, error=None, children=None):
    span = {"type": "agent", "name": name, "children": children or []}
    if error is not None:
        span["error"] = error
    return span


def _test_case(*children, root_type="agent"):
    case = LLMTestCase(input="do the task", actual_output="done")
    case._trace_dict = {
        "type": root_type,
        "name": "root",
        "children": list(children),
    }
    return case


class TestDelegationOutcomeMetric:
    """DelegationOutcomeMetric is deterministic, no API key needed."""

    def test_all_agents_succeed(self):
        metric = DelegationOutcomeMetric()
        metric.measure(
            _test_case(
                _agent_span("researcher"), _agent_span("writer"), root_type="x"
            )
        )
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_errored_sub_agent_scores_partially(self):
        metric = DelegationOutcomeMetric()
        metric.measure(
            _test_case(
                _agent_span("main"),
                _agent_span("researcher", error="Timeout"),
                root_type="x",
            )
        )
        assert metric.score == 0.5
        assert "researcher (1)" in metric.reason
        assert metric.is_successful() is False

    def test_no_agents_passes(self):
        metric = DelegationOutcomeMetric()
        metric.measure(_test_case(root_type="base"))
        assert metric.score == 1.0
        assert "No agents" in metric.reason

    def test_nested_agents_are_counted(self):
        inner = _agent_span("inner", error="boom")
        metric = DelegationOutcomeMetric()
        metric.measure(_test_case(_agent_span("outer", children=[inner])))
        assert metric.score == 0.5

    def test_ignore_recovered_failures(self):
        metric = DelegationOutcomeMetric(ignore_recovered_failures=True)
        metric.measure(
            _test_case(
                _agent_span("worker", error="503"),
                _agent_span("worker"),
                root_type="x",
            )
        )
        assert metric.score == 1.0
        assert "recovered" in metric.reason

    def test_missing_trace_fails(self):
        metric = DelegationOutcomeMetric()
        metric.measure(LLMTestCase(input="do it", actual_output="done"))
        assert metric.score == 0.0
        assert "No trace data" in metric.reason

    def test_strict_mode_zeroes_partial(self):
        metric = DelegationOutcomeMetric(strict_mode=True)
        metric.measure(
            _test_case(
                _agent_span("a"), _agent_span("b", error="boom"), root_type="x"
            )
        )
        assert metric.score == 0

    def test_requires_trace(self):
        assert DelegationOutcomeMetric().requires_trace is True

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = DelegationOutcomeMetric()
        score = await metric.a_measure(
            _test_case(
                _agent_span("a"), _agent_span("b", error="boom"), root_type="x"
            )
        )
        assert score == 0.5
