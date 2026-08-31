"""Unit tests for the `regex` toggle of PatternMatchMetric.

`PatternMatchMetric` is fully deterministic (no LLM / API key required), so
these tests run offline. They cover both sides of the contract:

* default behavior does not regress — `regex=True` keeps the historical
  regex semantics exactly;
* the new capability works — `regex=False` treats the pattern as a literal
  string, with every regex metacharacter inert.
"""

import pytest

from deepeval.metrics import PatternMatchMetric
from deepeval.test_case import LLMTestCase


def _test_case(actual_output: str) -> LLMTestCase:
    return LLMTestCase(
        input="dummy input",
        actual_output=actual_output,
        expected_output=actual_output,
    )


# ---------------------------------------------------------------------------
# Default behavior must not regress (regex=True)
# ---------------------------------------------------------------------------


def test_regex_mode_is_the_default():
    # A dot is a regex wildcard, so "a.b" matches "acb" in default mode.
    metric = PatternMatchMetric(pattern="a.b")
    assert metric.measure(_test_case("acb")) == 1.0
    assert metric.measure(_test_case("aXb")) == 1.0
    # The middle dot also matches a literal dot, so "a.b" matches too…
    assert metric.measure(_test_case("a.b")) == 1.0
    # …but exactly one character is required between "a" and "b".
    assert metric.measure(_test_case("ab")) == 0.0


def test_regex_mode_anchors_fullmatch_by_default():
    metric = PatternMatchMetric(pattern=r"^Hello")
    assert metric.measure(_test_case("Hello world")) == 0.0  # fullmatch
    assert metric.measure(_test_case("Hello")) == 1.0


def test_regex_mode_supports_character_classes():
    metric = PatternMatchMetric(pattern=r"\d{2}-\d{2}")
    assert metric.measure(_test_case("12-34")) == 1.0
    assert metric.measure(_test_case("12-3")) == 0.0


def test_regex_mode_ignore_case_still_applies():
    metric = PatternMatchMetric(pattern="hello", ignore_case=True)
    assert metric.measure(_test_case("HELLO")) == 1.0


# ---------------------------------------------------------------------------
# New capability: literal (regex=False) matching
# ---------------------------------------------------------------------------


def test_literal_mode_matches_metacharacters_verbatim():
    # In literal mode "C++" is plain text, not "C" followed by two "+" quantifiers.
    metric = PatternMatchMetric(pattern="C++", regex=False)
    assert metric.measure(_test_case("C++")) == 1.0
    assert metric.measure(_test_case("C+")) == 0.0
    assert metric.measure(_test_case("Cplusplus")) == 0.0


def test_literal_mode_dot_is_not_a_wildcard():
    metric = PatternMatchMetric(pattern="a.b", regex=False)
    assert metric.measure(_test_case("a.b")) == 1.0
    assert metric.measure(_test_case("acb")) == 0.0
    assert metric.measure(_test_case("aXb")) == 0.0


def test_literal_mode_dollar_and_parentheses_are_inert():
    metric = PatternMatchMetric(pattern="price: $5.00 (tax incl.)", regex=False)
    assert metric.measure(_test_case("price: $5.00 (tax incl.)")) == 1.0
    assert metric.measure(_test_case("price: 5.00 tax incl")) == 0.0


def test_literal_mode_with_ignore_case():
    metric = PatternMatchMetric(
        pattern="version 1.2.3", regex=False, ignore_case=True
    )
    assert metric.measure(_test_case("VERSION 1.2.3")) == 1.0
    assert metric.measure(_test_case("version 1.2.4")) == 0.0


def test_literal_mode_strips_pattern_like_regex_mode():
    # Whitespace around the pattern is trimmed in both modes.
    metric = PatternMatchMetric(pattern="  C++  ", regex=False)
    assert metric.measure(_test_case("C++")) == 1.0


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_non_string_pattern_raises_type_error():
    with pytest.raises(
        TypeError, match="`pattern` must be a string for the 'Pattern Match'"
    ):
        PatternMatchMetric(pattern=123)


def test_invalid_regex_still_raises_value_error_in_regex_mode():
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        PatternMatchMetric(pattern="[unclosed", regex=True)


def test_literal_mode_never_raises_invalid_regex():
    # Even a string that is an invalid regex compiles fine once escaped.
    metric = PatternMatchMetric(pattern="[unclosed", regex=False)
    assert metric.measure(_test_case("[unclosed")) == 1.0


def test_score_and_success_are_set_in_both_modes():
    metric = PatternMatchMetric(pattern="ok", regex=False)
    score = metric.measure(_test_case("ok"))
    assert score == 1.0
    assert metric.score == 1.0
    assert metric.success is True
    assert metric.reason is not None

    metric2 = PatternMatchMetric(pattern="ok", regex=False)
    assert metric2.measure(_test_case("not ok")) == 0.0
    assert metric2.success is False
