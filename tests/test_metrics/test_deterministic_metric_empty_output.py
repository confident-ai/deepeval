import pytest

from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import ExactMatchMetric, PatternMatchMetric
from deepeval.test_case import LLMTestCase


def test_exact_match_rejects_whitespace_actual_output():
    metric = ExactMatchMetric()
    test_case = LLMTestCase(
        input="question",
        actual_output=" \n\t ",
        expected_output="expected",
    )

    with pytest.raises(MissingTestCaseParamsError) as exc_info:
        metric.measure(test_case, _show_indicator=False)

    assert "actual_output" in str(exc_info.value)


def test_exact_match_rejects_whitespace_expected_output():
    metric = ExactMatchMetric()
    test_case = LLMTestCase(
        input="question",
        actual_output="actual",
        expected_output=" \n\t ",
    )

    with pytest.raises(MissingTestCaseParamsError) as exc_info:
        metric.measure(test_case, _show_indicator=False)

    assert "expected_output" in str(exc_info.value)


def test_pattern_match_rejects_whitespace_actual_output():
    metric = PatternMatchMetric(pattern=r"\s*")
    test_case = LLMTestCase(
        input="question",
        actual_output=" \n\t ",
    )

    with pytest.raises(MissingTestCaseParamsError) as exc_info:
        metric.measure(test_case, _show_indicator=False)

    assert "actual_output" in str(exc_info.value)
