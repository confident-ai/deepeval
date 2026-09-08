"""Whitespace and case behavior of the two deterministic matchers.

These need no API key: both metrics are pure string comparison.
"""

import pytest

from deepeval.metrics import ExactMatchMetric, PatternMatchMetric
from deepeval.test_case import LLMTestCase


@pytest.mark.parametrize(
    ("actual_output", "expected_output", "expected_score"),
    [
        ("act", "act", 1.0),
        ("act ", "act", 0.0),
        (" act", "act", 0.0),
        ("act\n", "act", 0.0),
        ("act", "act ", 0.0),
        ("ACT", "act", 0.0),
    ],
)
def test_exact_match_compares_complete_strings(
    actual_output, expected_output, expected_score
):
    metric = ExactMatchMetric()
    metric.measure(
        LLMTestCase(
            input="classify",
            actual_output=actual_output,
            expected_output=expected_output,
        ),
        _show_indicator=False,
    )

    assert metric.score == expected_score


@pytest.mark.parametrize(
    ("pattern", "actual_output", "ignore_case", "expected_score"),
    [
        ("act", "act", False, 1.0),
        ("act", " act ", False, 0.0),
        ("act", "act\n", False, 0.0),
        (r"\s*act\s*", " act ", False, 1.0),
        (" act ", "act", False, 0.0),
        ("act", "ACT", False, 0.0),
        ("act", "ACT", True, 1.0),
    ],
)
def test_pattern_match_uses_complete_pattern_and_output(
    pattern, actual_output, ignore_case, expected_score
):
    metric = PatternMatchMetric(pattern=pattern, ignore_case=ignore_case)
    metric.measure(
        LLMTestCase(input="classify", actual_output=actual_output),
        _show_indicator=False,
    )

    assert metric.score == expected_score
