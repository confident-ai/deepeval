"""Tests for the community KeywordCoverageMetric.

The metric is fully deterministic (pure string matching, no LLM), so these run
without an API key and cover passing, failing, and edge-case behavior.
"""

import pytest

from deepeval.metrics.community import KeywordCoverageMetric
from deepeval.test_case import LLMTestCase


def _case(actual_output: str) -> LLMTestCase:
    return LLMTestCase(input="q", actual_output=actual_output)


class TestKeywordCoverageMetric:
    def test_full_coverage_passes(self):
        metric = KeywordCoverageMetric(
            keywords=["refund", "30-day", "return address"]
        )
        metric.measure(
            _case(
                "You get a 30-day full refund; ship it to our return address."
            )
        )
        assert metric.score == 1.0
        assert metric.success is True

    def test_partial_coverage_gives_fractional_score(self):
        metric = KeywordCoverageMetric(
            keywords=["refund", "30-day", "return address"]
        )
        metric.measure(_case("You get a full refund."))
        assert metric.score == pytest.approx(1 / 3)
        assert metric.success is False
        assert "30-day" in metric.reason and "return address" in metric.reason

    def test_forbidden_term_hard_fails_even_with_full_coverage(self):
        metric = KeywordCoverageMetric(
            keywords=["refund"],
            forbidden=["INTERNAL-CODENAME"],
        )
        metric.measure(
            _case("Refund approved. (INTERNAL-CODENAME: project-halo)")
        )
        assert metric.score == 0.0
        assert metric.success is False
        assert "INTERNAL-CODENAME" in metric.reason

    def test_partial_threshold_can_pass(self):
        metric = KeywordCoverageMetric(
            keywords=["alpha", "bravo", "charlie", "delta"],
            whole_word=True,
            threshold=0.5,
        )
        metric.measure(_case("alpha and bravo only"))
        assert metric.score == pytest.approx(0.5)
        assert metric.success is True

    def test_ignore_case_default_true(self):
        metric = KeywordCoverageMetric(keywords=["Refund"])
        metric.measure(_case("we issued a REFUND"))
        assert metric.score == 1.0

    def test_case_sensitive_when_ignore_case_false(self):
        metric = KeywordCoverageMetric(keywords=["Refund"], ignore_case=False)
        metric.measure(_case("we issued a refund"))
        assert metric.score == 0.0

    def test_whole_word_avoids_substring_false_positive(self):
        substring = KeywordCoverageMetric(keywords=["cat"])
        substring.measure(_case("the category is pets"))
        assert substring.score == 1.0  # substring match (default)

        whole = KeywordCoverageMetric(keywords=["cat"], whole_word=True)
        whole.measure(_case("the category is pets"))
        assert whole.score == 0.0  # no standalone 'cat'

        whole.measure(_case("the cat is here"))
        assert whole.score == 1.0

    def test_empty_keywords_raises(self):
        with pytest.raises(ValueError):
            KeywordCoverageMetric(keywords=[])
        with pytest.raises(ValueError):
            KeywordCoverageMetric(keywords=["  ", ""])

    def test_name(self):
        assert (
            KeywordCoverageMetric(keywords=["x"]).__name__ == "Keyword Coverage"
        )
