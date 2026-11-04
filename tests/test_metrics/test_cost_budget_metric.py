import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import CostBudgetMetric
from deepeval.test_case import LLMTestCase


def _test_case(token_cost):
    return LLMTestCase(
        input="do the task",
        actual_output="done",
        token_cost=token_cost,
    )


class TestCostBudgetMetric:
    """CostBudgetMetric is deterministic, so these run without any API key."""

    def test_under_budget_passes(self):
        metric = CostBudgetMetric(max_token_cost=0.05)
        metric.measure(_test_case(0.01))
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_exactly_at_budget_passes(self):
        metric = CostBudgetMetric(max_token_cost=0.05)
        metric.measure(_test_case(0.05))
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_zero_cost_passes(self):
        metric = CostBudgetMetric(max_token_cost=0.05)
        metric.measure(_test_case(0.0))
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_twice_budget_scores_half(self):
        metric = CostBudgetMetric(max_token_cost=0.05)
        metric.measure(_test_case(0.10))
        assert metric.score == 0.5
        assert metric.is_successful() is False
        assert "exceeds" in metric.reason

    def test_partial_credit_with_threshold(self):
        # 2x budget -> 0.5; passes at threshold 0.4.
        metric = CostBudgetMetric(max_token_cost=0.05, threshold=0.4)
        metric.measure(_test_case(0.10))
        assert metric.score == 0.5
        assert metric.is_successful() is True

    def test_strict_mode_zeroes_partial_success(self):
        metric = CostBudgetMetric(max_token_cost=0.05, strict_mode=True)
        metric.measure(_test_case(0.10))
        assert metric.score == 0
        assert metric.is_successful() is False

    def test_requires_positive_budget(self):
        with pytest.raises(ValueError):
            CostBudgetMetric(max_token_cost=0)
        with pytest.raises(ValueError):
            CostBudgetMetric(max_token_cost=-0.01)

    def test_missing_token_cost_raises(self):
        metric = CostBudgetMetric(max_token_cost=0.05)
        test_case = LLMTestCase(input="do the task", actual_output="done")
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(test_case)

    def test_negative_token_cost_raises(self):
        metric = CostBudgetMetric(max_token_cost=0.05)
        with pytest.raises(ValueError):
            metric.measure(_test_case(-0.01))

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = CostBudgetMetric(max_token_cost=0.05)
        score = await metric.a_measure(_test_case(0.10))
        assert score == 0.5
        assert metric.is_successful() is False
