"""Tests for toxic
"""


import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics.bias_classifier import UnBiasedMetric, assert_unbiased
from deepeval.run_test import assert_test
from deepeval.types import LLMTestCaseParams

output = "Winners of the FIFA world cup were the French national football team"

metric = UnBiasedMetric(evaluation_params=[LLMTestCaseParams.INPUT])


def test_non_toxic():
    assert_test(
        test_cases=LLMTestCase(
            input="placeholder",
            actual_output=output,
        ),
        metrics=[metric],
    )


def test_non_toxic_2():
    with pytest.raises(AssertionError):
        assert_unbiased(
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            input="placeholder",
            actual_output="Left wing is evil.",
        )


def test_non_toxic_metric():
    metric = UnBiasedMetric(evaluation_params=[LLMTestCaseParams.INPUT])
    test_case = LLMTestCase(input="placeholder", actual_output=output)
    assert_test(test_case, [metric])


def test_bias_score():
    metric = UnBiasedMetric(
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ]
    )
    test_case = LLMTestCase(input="placeholder", actual_output=output)
    assert_test(test_case, [metric])
