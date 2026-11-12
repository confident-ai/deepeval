"""Deterministic, key-free tests for the community SetMatchMetric."""

import asyncio

import pytest

from deepeval.metrics.community import SetMatchMetric
from deepeval.test_case import LLMTestCase


def _case(actual: str, expected: str) -> LLMTestCase:
    return LLMTestCase(
        input="List the items.",
        actual_output=actual,
        expected_output=expected,
    )


def test_order_insensitive_exact_match():
    metric = SetMatchMetric()
    score = metric.measure(_case("banana, apple", "apple, banana"))
    assert score == 1.0
    assert metric.success is True


def test_case_insensitive_by_default():
    metric = SetMatchMetric()
    assert metric.measure(_case("Apple, BANANA", "apple, banana")) == 1.0


def test_case_sensitive_flag_distinguishes():
    metric = SetMatchMetric(mode="recall", case_sensitive=True)
    # "Apple" != "apple" under case sensitivity, so only "banana" matches.
    assert metric.measure(_case("Apple, banana", "apple, banana")) == 0.5


def test_whitespace_is_stripped():
    metric = SetMatchMetric()
    assert metric.measure(_case("  apple ,  banana ", "apple,banana")) == 1.0


def test_duplicates_do_not_change_set():
    metric = SetMatchMetric()
    assert metric.measure(_case("apple, apple, banana", "apple, banana")) == 1.0


def test_semicolon_and_newline_delimiters():
    metric = SetMatchMetric()
    assert (
        metric.measure(_case("apple; banana\ncherry", "apple, banana, cherry"))
        == 1.0
    )


def test_bullet_and_number_markers_stripped():
    metric = SetMatchMetric()
    actual = "- apple\n- banana\n- cherry"
    expected = "1. apple\n2. banana\n3. cherry"
    assert metric.measure(_case(actual, expected)) == 1.0


def test_json_array_parsing():
    metric = SetMatchMetric()
    assert metric.measure(_case('["apple", "banana"]', "banana, apple")) == 1.0


def test_json_array_parsing_can_be_disabled():
    metric = SetMatchMetric(parse_json_arrays=False)
    # With JSON parsing off the brackets/quotes stay part of the tokens, so
    # the sets do not line up and the score drops below 1.0.
    assert metric.measure(_case('["apple", "banana"]', "apple, banana")) < 1.0


def test_partial_overlap_f1():
    metric = SetMatchMetric()
    # expected {a,b,c,d}, actual {a,b,x}: matched 2.
    # recall=0.5, precision=2/3, f1=2*pr/(p+r).
    score = metric.measure(_case("a, b, x", "a, b, c, d"))
    assert score == pytest.approx(0.5714285714, abs=1e-6)


def test_mode_recall():
    metric = SetMatchMetric(mode="recall")
    assert metric.measure(_case("a, b, x", "a, b, c, d")) == 0.5


def test_mode_precision():
    metric = SetMatchMetric(mode="precision")
    assert metric.measure(_case("a, b, x", "a, b, c, d")) == pytest.approx(
        2 / 3
    )


def test_extra_items_lower_precision_not_recall():
    recall_metric = SetMatchMetric(mode="recall")
    precision_metric = SetMatchMetric(mode="precision")
    case = _case("apple, banana, cherry, extra", "apple, banana, cherry")
    assert recall_metric.measure(case) == 1.0
    assert precision_metric.measure(case) == pytest.approx(0.75)


def test_no_overlap_scores_zero():
    metric = SetMatchMetric()
    assert metric.measure(_case("x, y", "a, b")) == 0.0
    assert metric.success is False


def test_empty_expected_raises():
    metric = SetMatchMetric()
    with pytest.raises(ValueError, match="no items"):
        metric.measure(_case("apple, banana", "   "))


def test_actual_with_no_items_scores_zero():
    # An empty-string actual_output is rejected centrally by the framework, so
    # this exercises the "actual parses to no items" path with whitespace.
    metric = SetMatchMetric()
    assert metric.measure(_case("   ", "apple, banana")) == 0.0
    assert metric.success is False


def test_include_reason_false():
    metric = SetMatchMetric(include_reason=False)
    metric.measure(_case("a, b", "a, c"))
    assert metric.reason is None


def test_reason_lists_missing_and_extra():
    metric = SetMatchMetric()
    metric.measure(_case("apple, orange", "apple, banana"))
    assert "banana" in metric.reason  # missing from output
    assert "orange" in metric.reason  # extra in output


def test_threshold_controls_success():
    metric = SetMatchMetric(threshold=0.5)
    # f1 for {a,b,x} vs {a,b,c,d} is ~0.571, above 0.5.
    assert metric.measure(_case("a, b, x", "a, b, c, d")) >= 0.5
    assert metric.success is True


def test_strict_mode_forces_threshold_one():
    metric = SetMatchMetric(threshold=0.1, strict_mode=True)
    assert metric.threshold == 1.0
    metric.measure(_case("a, b, x", "a, b, c, d"))
    assert metric.success is False


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode"):
        SetMatchMetric(mode="jaccard")


def test_async_parity():
    metric = SetMatchMetric()
    case = _case("a, b, x", "a, b, c, d")
    sync_score = metric.measure(case)
    async_score = asyncio.run(metric.a_measure(case))
    assert sync_score == async_score
