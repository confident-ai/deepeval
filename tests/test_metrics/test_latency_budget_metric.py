import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import LatencyBudgetMetric
from deepeval.test_case import LLMTestCase


def _test_case(completion_time):
    return LLMTestCase(
        input="do the task",
        actual_output="done",
        completion_time=completion_time,
    )


class TestLatencyBudgetMetric:
    """LatencyBudgetMetric is deterministic, so these run without any API key."""

    def test_under_budget_passes(self):
        metric = LatencyBudgetMetric(max_completion_time=10.0)
        metric.measure(_test_case(5.0))
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_exactly_at_budget_passes(self):
        metric = LatencyBudgetMetric(max_completion_time=10.0)
        metric.measure(_test_case(10.0))
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_zero_completion_time_passes(self):
        metric = LatencyBudgetMetric(max_completion_time=10.0)
        metric.measure(_test_case(0.0))
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_twice_budget_scores_half(self):
        metric = LatencyBudgetMetric(max_completion_time=10.0)
        metric.measure(_test_case(20.0))
        assert metric.score == 0.5
        assert metric.is_successful() is False
        assert "exceeds" in metric.reason

    def test_partial_credit_with_threshold(self):
        # 2x budget -> 0.5; passes at threshold 0.4.
        metric = LatencyBudgetMetric(max_completion_time=10.0, threshold=0.4)
        metric.measure(_test_case(20.0))
        assert metric.score == 0.5
        assert metric.is_successful() is True

    def test_strict_mode_zeroes_partial_success(self):
        metric = LatencyBudgetMetric(max_completion_time=10.0, strict_mode=True)
        metric.measure(_test_case(20.0))
        assert metric.score == 0
        assert metric.is_successful() is False

    def test_requires_positive_budget(self):
        with pytest.raises(ValueError):
            LatencyBudgetMetric(max_completion_time=0)
        with pytest.raises(ValueError):
            LatencyBudgetMetric(max_completion_time=-1.0)

    def test_missing_completion_time_raises(self):
        metric = LatencyBudgetMetric(max_completion_time=10.0)
        test_case = LLMTestCase(input="do the task", actual_output="done")
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(test_case)

    def test_negative_completion_time_raises(self):
        metric = LatencyBudgetMetric(max_completion_time=10.0)
        with pytest.raises(ValueError):
            metric.measure(_test_case(-1.0))

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = LatencyBudgetMetric(max_completion_time=10.0)
        score = await metric.a_measure(_test_case(20.0))
        assert score == 0.5
        assert metric.is_successful() is False
