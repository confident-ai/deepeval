import math

import pytest

from deepeval.metrics.community import NumericMatchMetric
from deepeval.test_case import LLMTestCase


def _case(expected: str, actual: str) -> LLMTestCase:
    return LLMTestCase(
        input="What is the value?",
        actual_output=actual,
        expected_output=expected,
    )


class TestNumericMatchMetric:
    def test_thousands_separator_matches(self):
        metric = NumericMatchMetric()
        metric.measure(
            _case("The total is 1,200 units.", "The total is 1200 units.")
        )
        assert metric.score == 1.0
        assert metric.success is True

    def test_trailing_zero_decimal_matches(self):
        metric = NumericMatchMetric()
        metric.measure(_case("3", "The answer is 3.0"))
        assert metric.score == 1.0

    def test_scientific_notation_matches(self):
        metric = NumericMatchMetric()
        metric.measure(_case("1200", "About 1.2e3"))
        assert metric.score == 1.0

    def test_currency_symbol_stripped(self):
        metric = NumericMatchMetric()
        metric.measure(_case("$1,200.50", "1200.5 dollars"))
        assert metric.score == 1.0

    def test_negative_sign_respected(self):
        metric = NumericMatchMetric()
        metric.measure(_case("-5", "the change was 5"))
        assert metric.score == 0.0
        assert metric.success is False

    def test_percent_defaults_to_bare_number(self):
        metric = NumericMatchMetric()
        metric.measure(_case("12%", "12"))
        assert metric.score == 1.0

    def test_percent_as_fraction_option(self):
        metric = NumericMatchMetric(percent_as_fraction=True)
        metric.measure(_case("12%", "0.12"))
        assert metric.score == 1.0

    def test_magnitude_suffix_off_by_default(self):
        metric = NumericMatchMetric()
        metric.measure(_case("$1.2M", "1200000"))
        assert metric.score == 0.0

    def test_magnitude_suffix_on(self):
        metric = NumericMatchMetric(parse_magnitude_suffixes=True)
        metric.measure(_case("$1.2M", "1200000"))
        assert metric.score == 1.0

    def test_partial_recall_of_reference_numbers(self):
        metric = NumericMatchMetric(threshold=0.5)
        metric.measure(
            _case("The trip cost 100 and took 3 days.", "It cost 100.")
        )
        assert metric.score == 0.5
        assert metric.success is True
        assert "3" in metric.reason

    def test_extra_output_numbers_do_not_lower_recall(self):
        metric = NumericMatchMetric()
        metric.measure(_case("42", "The values were 7, 42, and 99."))
        assert metric.score == 1.0

    def test_duplicates_are_not_double_counted(self):
        metric = NumericMatchMetric()
        # Two reference 5s, but the output only provides one 5.
        metric.measure(_case("5 and 5", "just one 5 here"))
        assert metric.score == 0.5

    def test_absolute_tolerance(self):
        strict = NumericMatchMetric()
        strict.measure(_case("100", "the reading was 100.4"))
        assert strict.score == 0.0

        lenient = NumericMatchMetric(abs_tol=0.5)
        lenient.measure(_case("100", "the reading was 100.4"))
        assert lenient.score == 1.0

    def test_relative_tolerance(self):
        metric = NumericMatchMetric(rel_tol=0.01)
        metric.measure(_case("1000", "about 1005"))
        assert metric.score == 1.0

    def test_no_reference_number_is_unscorable(self):
        metric = NumericMatchMetric()
        with pytest.raises(ValueError, match="no numeric value"):
            metric.measure(_case("no numbers here", "still nothing"))

    def test_include_reason_false_suppresses_reason(self):
        metric = NumericMatchMetric(include_reason=False)
        metric.measure(_case("10", "10"))
        assert metric.reason is None

    def test_strict_mode_forces_full_match_threshold(self):
        metric = NumericMatchMetric(threshold=0.5, strict_mode=True)
        assert metric.threshold == 1.0
        metric.measure(_case("1 and 2", "only 1"))
        assert metric.score == 0.5
        assert metric.success is False

    @pytest.mark.asyncio
    async def test_async_matches_sync(self):
        metric = NumericMatchMetric()
        score = await metric.a_measure(_case("1,000", "1000"))
        assert score == 1.0
        assert metric.score == 1.0

    def test_extraction_handles_grouped_and_plain(self):
        metric = NumericMatchMetric()
        numbers = metric._extract_numbers("1,234,567 and 89 and 0.5")
        assert numbers == [1234567.0, 89.0, 0.5]
        assert math.isclose(numbers[-1], 0.5)
