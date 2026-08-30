"""Tests for the optional normalisation on ExactMatchMetric.

These cover the two directions that matter: with no options the metric behaves
exactly as before, and a genuinely different answer is never made to match.
"""
from deepeval.metrics import ExactMatchMetric
from deepeval.test_case import LLMTestCase


def _case(expected, actual):
    return LLMTestCase(input="q", actual_output=actual, expected_output=expected)


def test_default_behaviour_unchanged():
    m = ExactMatchMetric()
    assert m.measure(_case("15-minute", "15-minute"), _show_indicator=False) == 1.0
    # café in NFD vs NFC differs by bytes; default is still strict.
    assert m.measure(_case("café", "café"), _show_indicator=False) == 0.0


def test_normalize_unicode_fixes_visually_identical():
    m = ExactMatchMetric(normalize_unicode=True)
    assert m.measure(_case("café", "café"), _show_indicator=False) == 1.0
    assert m.measure(_case("ﬁle", "file"), _show_indicator=False) == 1.0


def test_ignore_case_and_punctuation():
    assert ExactMatchMetric(ignore_case=True).measure(
        _case("Paris", "paris"), _show_indicator=False) == 1.0
    assert ExactMatchMetric(ignore_punctuation=True).measure(
        _case("Yes.", "Yes"), _show_indicator=False) == 1.0


def test_a_real_difference_never_matches():
    # The point of exact match survives: normalisation must not relax wording.
    m = ExactMatchMetric(ignore_case=True, ignore_punctuation=True,
                         normalize_unicode=True)
    assert m.measure(_case("$8.540", "$9.540"), _show_indicator=False) == 0.0
