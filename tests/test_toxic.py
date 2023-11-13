"""Tests for toxic
"""


import pytest
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.toxic_classifier import NonToxicMetric, assert_non_toxic
from deepeval.evaluator import assert_test

output = "Winners of the FIFA world cup were the French national football team"


def test_non_toxic():
    assert_non_toxic(
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        input="Placeholder",
        actual_output="Who won the FIFA World Cup in 2018?",
    )


def test_non_toxic_2():
    with pytest.raises(AssertionError):
        assert_non_toxic(
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            input="Placeholder",
            actual_output="You are stupid!",
        )


def test_non_toxic_metric():
    metric = NonToxicMetric(evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT])
    test_case = LLMTestCase(input="placeholder", actual_output=output)
    assert_test(test_case, [metric])
