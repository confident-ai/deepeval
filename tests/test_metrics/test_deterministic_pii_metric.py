import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics.community import DeterministicPIIMetric
from deepeval.test_case import LLMTestCase


def _test_case(actual_output):
    return LLMTestCase(
        input="What is the customer's contact info?",
        actual_output=actual_output,
    )


class TestDeterministicPIIMetric:
    """DeterministicPIIMetric is deterministic, so these run without an API key."""

    def test_clean_output_passes(self):
        metric = DeterministicPIIMetric()
        metric.measure(_test_case("Your refund is being processed."))
        assert metric.score == 1.0
        assert metric.is_successful() is True

    @pytest.mark.parametrize(
        "entity, text",
        [
            ("email", "Reach them at jane.doe@example.com today."),
            ("phone", "Call (415) 555-0132 for support."),
            ("phone", "Call +1 415.555.0132 for support."),
            ("ssn", "Their SSN is 123-45-6789."),
            ("credit_card", "Card on file: 4111 1111 1111 1111."),
            ("credit_card", "Card on file: 4111-1111-1111-1111."),
            ("ip_address", "The login came from 192.168.1.20."),
        ],
    )
    def test_detects_built_in_entity(self, entity, text):
        metric = DeterministicPIIMetric()
        metric.measure(_test_case(text))
        assert metric.score == 0.0
        assert metric.is_successful() is False
        assert entity in metric.reason

    def test_card_failing_luhn_is_not_flagged(self):
        metric = DeterministicPIIMetric(entities=["credit_card"])
        metric.measure(_test_case("Order number 4111 1111 1111 1112."))
        assert metric.score == 1.0

    @pytest.mark.parametrize(
        "text",
        [
            "The SSN 000-12-3456 is invalid.",
            "Version 999.1.2.3 was released.",
            "Invalid address 256.1.1.1 was rejected.",
            "Build 10.0.0.1.5 is a version string, not an address.",
        ],
    )
    def test_invalid_values_are_not_flagged(self, text):
        metric = DeterministicPIIMetric(entities=["ssn", "ip_address"])
        metric.measure(_test_case(text))
        assert metric.score == 1.0

    def test_reason_never_contains_matched_values(self):
        metric = DeterministicPIIMetric()
        metric.measure(
            _test_case("Email jane.doe@example.com, SSN 123-45-6789.")
        )
        assert metric.score == 0.0
        assert "jane.doe@example.com" not in metric.reason
        assert "123-45-6789" not in metric.reason
        assert "email (1)" in metric.reason
        assert "ssn (1)" in metric.reason

    def test_counts_multiple_matches(self):
        metric = DeterministicPIIMetric(entities=["email"])
        metric.measure(_test_case("a@example.com and b@example.org"))
        assert "email (2)" in metric.reason

    def test_entities_subset_ignores_others(self):
        metric = DeterministicPIIMetric(entities=["ssn"])
        metric.measure(_test_case("Reach them at jane.doe@example.com."))
        assert metric.score == 1.0
        assert metric.is_successful() is True

    def test_custom_pattern_detected(self):
        metric = DeterministicPIIMetric(
            entities=[], custom_patterns={"mrn": r"\bMRN-\d{6}\b"}
        )
        metric.measure(_test_case("Patient record MRN-004521 was updated."))
        assert metric.score == 0.0
        assert "mrn (1)" in metric.reason

    def test_include_reason_false(self):
        metric = DeterministicPIIMetric(include_reason=False)
        metric.measure(_test_case("Their SSN is 123-45-6789."))
        assert metric.score == 0.0
        assert metric.reason is None

    def test_unknown_entity_raises(self):
        with pytest.raises(ValueError):
            DeterministicPIIMetric(entities=["passport"])

    def test_custom_pattern_name_collision_raises(self):
        with pytest.raises(ValueError):
            DeterministicPIIMetric(custom_patterns={"email": r"@"})

    def test_invalid_custom_regex_raises(self):
        with pytest.raises(ValueError):
            DeterministicPIIMetric(custom_patterns={"bad": r"("})

    def test_nothing_to_check_raises(self):
        with pytest.raises(ValueError):
            DeterministicPIIMetric(entities=[])

    def test_missing_actual_output_raises(self):
        metric = DeterministicPIIMetric()
        with pytest.raises(MissingTestCaseParamsError):
            metric.measure(LLMTestCase(input="What is the contact info?"))

    @pytest.mark.asyncio
    async def test_async_measure_matches_sync(self):
        metric = DeterministicPIIMetric()
        score = await metric.a_measure(_test_case("SSN 123-45-6789"))
        assert score == 0.0
        assert metric.is_successful() is False
