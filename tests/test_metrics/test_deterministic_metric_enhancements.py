"""Deterministic (non-LLM) metric engineering enhancements.

These tests exercise the engineering additions that were previously
unhandled "edge scenarios" for the deterministic metrics:

* ``ExactMatchMetric(normalize=True)`` — case / internal-whitespace /
  Unicode-NFKC normalisation so semantically-equal outputs aren't flagged
  as false failures, while the default ``normalize=False`` keeps the exact
  historical behaviour.
* ``PatternMatchMetric(match_mode=...)`` — search / match / fullmatch
  semantics (was hard-coded to fullmatch), plus a loud error for an empty
  pattern that used to silently match everything.

All tests are offline: they call ``measure()`` directly and need no API key.
"""

from __future__ import annotations

import pytest

from deepeval.metrics import ExactMatchMetric, PatternMatchMetric
from deepeval.metrics.exact_match.exact_match import ExactMatchMetric as EMM
from deepeval.metrics.utils import MissingTestCaseParamsError
from deepeval.test_case import LLMTestCase


def _em(expected: str, actual: str, **kwargs) -> LLMTestCase:
    return LLMTestCase(
        input="a test input",
        expected_output=expected,
        actual_output=actual,
        **kwargs,
    )


def _pm(actual: str, **kwargs) -> LLMTestCase:
    return LLMTestCase(input="a test input", actual_output=actual, **kwargs)


# ---------------------------------------------------------------------------
# 1. ExactMatchMetric — default behaviour must not regress.
# ---------------------------------------------------------------------------


class TestExactMatchDefaultRegression:
    def test_exact_equal_is_pass(self):
        m = ExactMatchMetric()
        m.measure(_em("hello", "hello"))
        assert m.score == 1.0
        assert m.success is True

    def test_different_is_fail(self):
        m = ExactMatchMetric()
        m.measure(_em("hello", "world"))
        assert m.score == 0.0
        assert m.success is False

    def test_edge_whitespace_strip_still_applies(self):
        # `.strip()` is applied regardless of `normalize`; this was the
        # pre-existing behaviour and must be preserved.
        m = ExactMatchMetric()
        m.measure(_em("hello", "  hello  "))
        assert m.score == 1.0

    def test_case_sensitivity_default_is_exact(self):
        # Default normalize=False keeps raw == comparison.
        m = ExactMatchMetric()
        m.measure(_em("Hello", "hello"))
        assert m.score == 0.0

    def test_dead_precision_recall_f1_fields_removed(self):
        # Previously the metric assigned self.precision/recall/f1 but they
        # were never computed or consumed anywhere. They should no longer be
        # set (and were never real fields on BaseMetric).
        m = ExactMatchMetric()
        m.measure(_em("x", "x"))
        assert not hasattr(m, "precision")
        assert not hasattr(m, "recall")
        assert not hasattr(m, "f1")

    def test_empty_expected_output_rejected(self):
        # Symmetric to actual_output: an empty reference answer is a config
        # bug for a deterministic metric, not a meaningful test.
        with pytest.raises(MissingTestCaseParamsError, match="expected_output"):
            m = ExactMatchMetric()
            m.measure(_em("", "hello"))


# ---------------------------------------------------------------------------
# 2. ExactMatchMetric — the new normalize=True capability.
# ---------------------------------------------------------------------------


class TestExactMatchNormalize:
    def test_normalize_true_case_insensitive(self):
        m = ExactMatchMetric(normalize=True)
        m.measure(_em("Hello World", "HELLO world"))
        assert m.score == 1.0

    def test_normalize_collapses_internal_whitespace(self):
        m = ExactMatchMetric(normalize=True)
        m.measure(
            _em("refund at no extra cost", "refund  at\nno\textra   cost")
        )
        assert m.score == 1.0

    def test_normalize_unicode_nfkc_fullwidth(self):
        # Full-width forms ("ａｆｆｉｒｍ") are normalised to half-width.
        m = ExactMatchMetric(normalize=True)
        m.measure(_em("affirm", "ａｆｆｉｒｍ"))
        assert m.score == 1.0

    def test_normalize_casefold_double_s(self):
        # casefold() is more aggressive than lower(): "Straße" → "strasse".
        m = ExactMatchMetric(normalize=True)
        m.measure(_em("strasse", "Straße"))
        assert m.score == 1.0

    def test_normalize_is_idempotent(self):
        m = ExactMatchMetric(normalize=True)
        once = m._normalize_text("HELLO   WORLD\t\n")
        twice = m._normalize_text(once)
        assert once == twice

    def test_normalize_does_not_equalize_semantically_different(self):
        # Normalisation fixes *formatting* differences, not meaning.
        m = ExactMatchMetric(normalize=True)
        m.measure(_em("hello world", "hello there"))
        assert m.score == 0.0

    def test_normalize_static_helper_unit(self):
        assert EMM._normalize_text("  A ｂ \t C  ") == "a b c"
        assert EMM._normalize_text("ABCD") == "abcd"


# ---------------------------------------------------------------------------
# 3. PatternMatchMetric — default fullmatch regression + empty-pattern guard.
# ---------------------------------------------------------------------------


class TestPatternMatchDefaultRegression:
    PAT = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"

    def test_full_fullmatch_is_pass(self):
        m = PatternMatchMetric(pattern=self.PAT)
        m.measure(_pm("2024-12-31"))
        assert m.score == 1.0

    def test_partial_does_not_fullmatch_by_default(self):
        # Default match_mode="fullmatch": embedded date in longer text is a fail.
        m = PatternMatchMetric(pattern=self.PAT)
        m.measure(_pm("the date is 2024-12-31."))
        assert m.score == 0.0

    def test_empty_pattern_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            PatternMatchMetric(pattern="  ")

    def test_invalid_regex_forwarded(self):
        with pytest.raises(ValueError, match="Invalid regex pattern"):
            PatternMatchMetric(pattern="[unclosed")

    def test_invalid_match_mode_rejected(self):
        with pytest.raises(ValueError, match="match_mode"):
            PatternMatchMetric(pattern="x", match_mode="nope")


# ---------------------------------------------------------------------------
# 4. PatternMatchMetric — the new match_mode capability.
# ---------------------------------------------------------------------------


class TestPatternMatchMatchMode:
    PAT = r"\d{4}-\d{2}-\d{2}"

    def test_search_finds_substring_anywhere(self):
        m = PatternMatchMetric(pattern=self.PAT, match_mode="search")
        m.measure(_pm("the date is 2024-12-31, thanks"))
        assert m.score == 1.0

    def test_search_no_match_is_fail(self):
        m = PatternMatchMetric(pattern=self.PAT, match_mode="search")
        m.measure(_pm("no date here"))
        assert m.score == 0.0

    def test_match_anchors_at_start_only(self):
        # `match` anchors at position 0 but does not require full consumption,
        # so a leading match within a longer line still passes.
        m = PatternMatchMetric(pattern=self.PAT, match_mode="match")
        m.measure(_pm("2024-12-31 then trailing text"))
        assert m.score == 1.0

    def test_match_rejects_not_at_start(self):
        m = PatternMatchMetric(pattern=self.PAT, match_mode="match")
        m.measure(_pm("leading text 2024-12-31"))
        assert m.score == 0.0

    def test_modes_are_mutually_distinct(self):
        # Same pattern + same input, three modes give different verdicts.
        # PAT="2024"; input "year 2024 end":
        #   search  -> the sub-string "2024" is found in the middle -> pass
        #   match   -> the string must *start* at "2024", it starts at "year" -> fail
        #   fullmatch -> the whole string must equal "2024" -> fail
        case = _pm("year 2024 end")
        for mode in ("match", "fullmatch"):
            m = PatternMatchMetric(pattern=r"2024", match_mode=mode)
            m.measure(case)
            assert m.score == 0.0, mode
        m = PatternMatchMetric(pattern=r"2024", match_mode="search")
        m.measure(case)
        assert m.score == 1.0

    def test_match_vs_fullmatch_distinction(self):
        # A prefix-only match passes `match` but fails `fullmatch`.
        case = _pm("2024-12-31 then trailing text")
        m = PatternMatchMetric(pattern=r"2024", match_mode="match")
        m.measure(case)
        assert m.score == 1.0
        m = PatternMatchMetric(pattern=r"2024", match_mode="fullmatch")
        m.measure(case)
        assert m.score == 0.0

    def test_ignore_case_mode_combination(self):
        m = PatternMatchMetric(
            pattern=r"refund policy", ignore_case=True, match_mode="search"
        )
        m.measure(_pm("See our REFUND POLICY section for details."))
        assert m.score == 1.0
