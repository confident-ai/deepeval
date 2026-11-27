import pytest

from deepeval.metrics.community import StreamingLatencyMetric
from deepeval.test_case import LLMTestCase


def _test_case(intervals=None, trace=True):
    case = LLMTestCase(input="Stream this.", actual_output="Streamed.")
    if trace:
        children = []
        if intervals is not None:
            children.append(
                {
                    "type": "llm",
                    "name": "llm",
                    "children": [],
                    "token_intervals": intervals,
                }
            )
        case._trace_dict = {
            "type": "agent",
            "name": "agent",
            "children": children,
        }
    return case


class TestStreamingLatencyMetric:
    """StreamingLatencyMetric is deterministic, no API key needed."""

    def test_ttft_within_budget_passes(self):
        metric = StreamingLatencyMetric(max_ttft=0.5)
        metric.measure(
            _test_case(),
            token_times=[0.2, 0.3, 0.4],
            start_time=0.0,
        )
        assert metric.score == 1.0
        assert metric.score_breakdown["ttft_s"] == pytest.approx(0.2)
        assert metric.is_successful() is True

    def test_ttft_over_budget_scores_proportionally(self):
        metric = StreamingLatencyMetric(max_ttft=0.1)
        metric.measure(
            _test_case(),
            token_times=[0.2, 0.3, 0.4],
            start_time=0.0,
        )
        assert metric.score == pytest.approx(0.5)
        assert metric.is_successful() is False

    def test_throughput_budget(self):
        metric = StreamingLatencyMetric(min_tokens_per_sec=2.0)
        metric.measure(_test_case(), token_times=[0.0, 1.0, 2.0, 3.0])
        assert metric.score_breakdown["tokens_per_sec"] == pytest.approx(1.0)
        assert metric.score == pytest.approx(0.5)

    def test_tbt_budget(self):
        metric = StreamingLatencyMetric(max_tbt_mean=0.2)
        metric.measure(_test_case(), token_times=[0.0, 0.1, 0.2, 0.3])
        assert metric.score_breakdown["tbt_mean_s"] == pytest.approx(0.1)
        assert metric.score == 1.0

    def test_worst_budget_wins(self):
        metric = StreamingLatencyMetric(max_ttft=5.0, min_tokens_per_sec=2.0)
        metric.measure(
            _test_case(),
            token_times=[0.0, 1.0, 2.0, 3.0],
            start_time=-0.1,
        )
        assert metric.score == pytest.approx(0.5)

    def test_trace_intervals_are_used(self):
        metric = StreamingLatencyMetric(max_tbt_mean=0.5)
        metric.measure(_test_case({0.0: "a", 0.1: "b", 0.2: "c"}))
        assert metric.score == 1.0
        assert metric.score_breakdown["n_tokens"] == 3.0

    def test_iso_string_intervals_parse(self):
        metric = StreamingLatencyMetric(max_tbt_mean=5.0)
        metric.measure(
            _test_case(
                {
                    "2026-01-01T00:00:00+00:00": "a",
                    "2026-01-01T00:00:01+00:00": "b",
                }
            )
        )
        assert metric.score == 1.0

    def test_unmeasurable_ttft_with_budget_fails(self):
        metric = StreamingLatencyMetric(max_ttft=5.0)
        metric.measure(_test_case(), token_times=[0.0, 1.0, 2.0])
        assert metric.score == 0.0
        assert "not measurable" in metric.reason

    def test_missing_timings_fail(self):
        metric = StreamingLatencyMetric(min_tokens_per_sec=1.0)
        metric.measure(_test_case(trace=False))
        assert metric.score == 0.0
        assert "token" in metric.reason

    def test_no_budgets_rejected(self):
        with pytest.raises(ValueError):
            StreamingLatencyMetric()

    def test_bad_budget_rejected(self):
        with pytest.raises(ValueError):
            StreamingLatencyMetric(max_ttft=0)

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = StreamingLatencyMetric(min_tokens_per_sec=10.0)
        score = await metric.a_measure(
            _test_case(), token_times=[0.0, 1.0, 2.0]
        )
        assert score == pytest.approx(0.1)
