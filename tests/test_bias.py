"""Tests for toxic
"""


import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics.bias_classifier import UnBiasedMetric, assert_unbiased
from deepeval.run_test import assert_test

output = "Winners of the FIFA world cup were the French national football team"

metric = UnBiasedMetric()


def test_non_toxic():
    assert_test(
        test_cases=LLMTestCase(
            output=output,
        ),
        metrics=[metric],
    )


def test_non_toxic_2():
    with pytest.raises(AssertionError):
        assert_unbiased(
            text="Left wing is evil.",
        )


def test_non_toxic_metric():
    metric = UnBiasedMetric()
    test_case = LLMTestCase(output=output)
    assert_test(test_case, [metric])
    assert_test(test_case, [metric])


def test_bias_score():
    metric = UnBiasedMetric()
    test_case = LLMTestCase(output=output)
    assert_test(test_case, [metric])
