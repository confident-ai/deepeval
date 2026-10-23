import pytest

from deepeval.metrics.community import ToolRetryThrashMetric
from deepeval.test_case import LLMTestCase


def _tool_span(name, args=None):
    return {
        "type": "tool",
        "name": name,
        "input": args or {},
        "children": [],
    }


def _test_case(*tool_spans, trace=True):
    case = LLMTestCase(input="do the task", actual_output="done")
    if trace:
        case._trace_dict = {
            "type": "agent",
            "name": "agent",
            "children": list(tool_spans),
        }
    return case


class TestToolRetryThrashMetric:
    """ToolRetryThrashMetric is deterministic, no API key needed."""

    def test_clean_sequence_passes(self):
        metric = ToolRetryThrashMetric()
        metric.measure(
            _test_case(
                _tool_span("search", {"q": "a"}),
                _tool_span("fetch", {"u": "b"}),
                _tool_span("reply"),
            )
        )
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_repeated_calls_degrade(self):
        metric = ToolRetryThrashMetric()
        metric.measure(_test_case(*[_tool_span("search", {"q": "a"})] * 3))
        assert metric.score_breakdown["retry_score"] == 0.5
        assert metric.score_breakdown["thrash_score"] == 1.0
        assert metric.score == 0.75

    def test_heavy_repetition_fails_retry(self):
        metric = ToolRetryThrashMetric()
        metric.measure(_test_case(*[_tool_span("search", {"q": "a"})] * 6))
        assert metric.score_breakdown["retry_score"] == 0.0
        assert metric.score == 0.5

    def test_same_tool_new_args_is_not_a_retry(self):
        metric = ToolRetryThrashMetric()
        metric.measure(
            _test_case(
                _tool_span("search", {"q": "a"}),
                _tool_span("search", {"q": "b"}),
                _tool_span("search", {"q": "c"}),
            )
        )
        assert metric.score == 1.0

    def test_flip_flop_degrades(self):
        metric = ToolRetryThrashMetric()
        metric.measure(
            _test_case(_tool_span("a"), _tool_span("b"), _tool_span("a"))
        )
        assert metric.score_breakdown["thrash_score"] == 0.5
        assert metric.score == 0.75

    def test_sustained_thrash_fails(self):
        metric = ToolRetryThrashMetric()
        metric.measure(
            _test_case(
                _tool_span("a", {"i": 1}),
                _tool_span("b", {"i": 2}),
                _tool_span("a", {"i": 3}),
                _tool_span("b", {"i": 4}),
                _tool_span("a", {"i": 5}),
            )
        )
        assert metric.score_breakdown["retry_score"] == 1.0
        assert metric.score_breakdown["thrash_score"] == 0.0
        assert metric.score == 0.5

    def test_no_tools_passes(self):
        metric = ToolRetryThrashMetric()
        metric.measure(_test_case())
        assert metric.score == 1.0

    def test_missing_trace_fails(self):
        metric = ToolRetryThrashMetric()
        metric.measure(_test_case(trace=False))
        assert metric.score == 0.0
        assert "No trace data" in metric.reason

    def test_bad_thresholds_rejected(self):
        with pytest.raises(ValueError):
            ToolRetryThrashMetric(repetition_threshold=1)
        with pytest.raises(ValueError):
            ToolRetryThrashMetric(alternation_threshold=0)

    def test_strict_mode_zeroes_partial(self):
        metric = ToolRetryThrashMetric(strict_mode=True)
        metric.measure(_test_case(*[_tool_span("search", {"q": "a"})] * 3))
        assert metric.score == 0

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = ToolRetryThrashMetric()
        score = await metric.a_measure(
            _test_case(_tool_span("a"), _tool_span("b"), _tool_span("a"))
        )
        assert score == 0.75
