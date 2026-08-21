"""Regression tests for #3057 (verdict whitespace) and #3050 (empty-output guard).

These tests do NOT call any LLM — they directly exercise the scoring and
validation logic by injecting pre-built verdict objects or test cases,
verifying that whitespace variants no longer corrupt results.
"""

import pytest
import os

# Set a fake key so metric constructors don't raise during import.
# These tests never call any LLM.
os.environ["OPENAI_API_KEY"] = (
    os.environ.get("OPENAI_API_KEY") or "sk-fake-for-unit-tests-only"
)


from deepeval.metrics import ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.metrics.contextual_recall.schema import ContextualRecallVerdict
from deepeval.metrics.contextual_relevancy.schema import (
    ContextualRelevancyVerdict,
    ContextualRelevancyVerdicts,
)
from deepeval.test_case import LLMTestCase

# ===========================================================================
# #3057 — verdict.strip() before comparison
# ===========================================================================


class TestVerdictWhitespaceStripping:
    """Trailing/leading whitespace in verdict strings must not change scores."""

    @pytest.mark.parametrize(
        "padded_yes",
        ["yes", "yes ", " yes", " yes ", "Yes ", " YES"],
        ids=lambda v: repr(v),
    )
    def test_contextual_recall_accepts_padded_yes(self, padded_yes):
        """ContextualRecall._calculate_score treats padded 'yes' as affirmative."""
        metric = ContextualRecallMetric(threshold=0, include_reason=False)
        metric.verdicts = [
            ContextualRecallVerdict(verdict=padded_yes, reason="ok"),
            ContextualRecallVerdict(verdict="no", reason="not ok"),
        ]
        score = metric._calculate_score()
        assert score == pytest.approx(
            0.5
        ), f"verdict={padded_yes!r} should count as 'yes', expected 0.5, got {score}"

    @pytest.mark.parametrize(
        "padded_no",
        ["no", "no ", " no", " no ", "No ", " NO"],
        ids=lambda v: repr(v),
    )
    def test_contextual_relevancy_accepts_padded_no(self, padded_no):
        """ContextualRelevancy._calculate_score treats padded 'no' as negative."""
        metric = ContextualRelevancyMetric(threshold=0, include_reason=False)
        metric.verdicts_list = [
            ContextualRelevancyVerdicts(
                verdicts=[
                    ContextualRelevancyVerdict(
                        verdict="yes", reason="relevant", statement="A"
                    ),
                    ContextualRelevancyVerdict(
                        verdict=padded_no, reason="irrelevant", statement="B"
                    ),
                ]
            )
        ]
        score = metric._calculate_score()
        # 1 yes out of 2 total -> 0.5
        assert score == pytest.approx(
            0.5
        ), f"verdict={padded_no!r} should count as 'no', expected 0.5, got {score}"

    def test_contextual_recall_score_unchanged_for_clean_verdicts(self):
        """Verify no regression: clean verdicts still produce correct score."""
        metric = ContextualRecallMetric(threshold=0, include_reason=False)
        metric.verdicts = [
            ContextualRecallVerdict(verdict="yes", reason="a"),
            ContextualRecallVerdict(verdict="yes", reason="b"),
            ContextualRecallVerdict(verdict="no", reason="c"),
            ContextualRecallVerdict(verdict="yes", reason="d"),
        ]
        score = metric._calculate_score()
        assert score == pytest.approx(0.75)

    def test_contextual_relevancy_score_unchanged_for_clean_verdicts(self):
        """Verify no regression: clean verdicts still produce correct score."""
        metric = ContextualRelevancyMetric(threshold=0, include_reason=False)
        metric.verdicts_list = [
            ContextualRelevancyVerdicts(
                verdicts=[
                    ContextualRelevancyVerdict(
                        verdict="yes", reason="r", statement="s1"
                    ),
                    ContextualRelevancyVerdict(
                        verdict="yes", reason="r", statement="s2"
                    ),
                    ContextualRelevancyVerdict(
                        verdict="no", reason="r", statement="s3"
                    ),
                ]
            )
        ]
        score = metric._calculate_score()
        # 2 yes out of 3
        assert score == pytest.approx(2 / 3)

    def test_mixed_padding_produces_correct_count(self):
        """Mix of padded and clean verdicts: 3/4 yes -> 0.75."""
        metric = ContextualRecallMetric(threshold=0, include_reason=False)
        metric.verdicts = [
            ContextualRecallVerdict(verdict="yes", reason="clean"),
            ContextualRecallVerdict(verdict="yes ", reason="trailing space"),
            ContextualRecallVerdict(verdict=" yes", reason="leading space"),
            ContextualRecallVerdict(verdict="no", reason="negative"),
        ]
        score = metric._calculate_score()
        assert score == pytest.approx(0.75)
