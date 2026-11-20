import pytest

from deepeval.metrics.community import ContextWindowBudgetMetric
from deepeval.test_case import LLMTestCase


def _llm_span(input_tokens):
    return {
        "type": "llm",
        "name": "llm",
        "input_token_count": input_tokens,
        "children": [],
    }


def _test_case(**kwargs):
    return LLMTestCase(
        input="Summarize this long document please.",
        actual_output="Summary.",
        **kwargs,
    )


class TestContextWindowBudgetMetric:
    """ContextWindowBudgetMetric is deterministic, no API key needed."""

    def test_within_budget_passes(self):
        metric = ContextWindowBudgetMetric(max_window_tokens=1000)
        metric.measure(_test_case(input_token_count=500))
        assert metric.score == 1.0
        assert metric.score_breakdown["utilization"] == 0.5
        assert metric.is_successful() is True

    def test_over_budget_scores_proportionally(self):
        metric = ContextWindowBudgetMetric(max_window_tokens=1000)
        metric.measure(_test_case(input_token_count=1800))
        assert metric.score == pytest.approx(900 / 1800)
        assert metric.is_successful() is False
        assert "above" in metric.reason

    def test_trace_counts_are_summed(self):
        case = _test_case()
        case._trace_dict = {
            "type": "agent",
            "name": "agent",
            "children": [_llm_span(300), _llm_span(200)],
        }
        metric = ContextWindowBudgetMetric(max_window_tokens=1000)
        metric.measure(case)
        assert metric.score == 1.0
        assert metric.score_breakdown["prompt_tokens"] == 500.0

    def test_missing_counts_fall_back_to_estimate(self):
        metric = ContextWindowBudgetMetric(max_window_tokens=100000)
        metric.measure(_test_case())
        assert metric.score == 1.0
        assert "estimated" in metric.reason

    def test_invalid_window_rejected(self):
        with pytest.raises(ValueError):
            ContextWindowBudgetMetric(max_window_tokens=0)

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            ContextWindowBudgetMetric(
                max_window_tokens=1000, warn_threshold=1.5
            )

    def test_strict_mode_zeroes_partial(self):
        metric = ContextWindowBudgetMetric(
            max_window_tokens=1000, strict_mode=True
        )
        metric.measure(_test_case(input_token_count=950))
        assert metric.score == 0

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = ContextWindowBudgetMetric(max_window_tokens=1000)
        score = await metric.a_measure(_test_case(input_token_count=1800))
        assert score == pytest.approx(0.5)
