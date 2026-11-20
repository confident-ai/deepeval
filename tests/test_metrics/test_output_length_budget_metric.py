import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import OutputLengthBudgetMetric
from deepeval.test_case import LLMTestCase


def _test_case(actual_output):
    return LLMTestCase(
        input="Say something.",
        actual_output=actual_output,
    )


class TestOutputLengthBudgetMetric:
    """OutputLengthBudgetMetric is deterministic, no API key needed."""

    def test_within_range_passes(self):
        metric = OutputLengthBudgetMetric(min_length=2, max_length=10)
        metric.measure(_test_case("four score words here yes"))
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_too_long_scores_proportionally(self):
        metric = OutputLengthBudgetMetric(max_length=4)
        metric.measure(_test_case("one two three four five six"))
        assert metric.score == pytest.approx(4 / 6)
        assert metric.is_successful() is False

    def test_too_short_scores_proportionally(self):
        metric = OutputLengthBudgetMetric(min_length=4)
        metric.measure(_test_case("just two"))
        assert metric.score == pytest.approx(2 / 4)

    def test_chars_unit(self):
        metric = OutputLengthBudgetMetric(
            max_length=5, unit="chars", threshold=0.5
        )
        metric.measure(_test_case("12345678"))
        assert metric.score == pytest.approx(5 / 8)
        assert metric.score_breakdown["length"] == 8.0

    def test_no_bounds_rejected(self):
        with pytest.raises(ValueError):
            OutputLengthBudgetMetric()

    def test_bad_unit_rejected(self):
        with pytest.raises(ValueError):
            OutputLengthBudgetMetric(max_length=5, unit="tokens")

    def test_min_above_max_rejected(self):
        with pytest.raises(ValueError):
            OutputLengthBudgetMetric(min_length=10, max_length=5)

    def test_strict_mode_zeroes_partial(self):
        metric = OutputLengthBudgetMetric(max_length=2, strict_mode=True)
        metric.measure(_test_case("one two three"))
        assert metric.score == 0

    def test_missing_actual_output_raises(self):
        metric = OutputLengthBudgetMetric(max_length=5)
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(LLMTestCase(input="Say something."))

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = OutputLengthBudgetMetric(max_length=2)
        score = await metric.a_measure(_test_case("one two three"))
        assert score == pytest.approx(2 / 3)
