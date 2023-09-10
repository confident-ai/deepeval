"""Tests for toxic
"""


import pytest
from deepeval.test_case import LLMTestCase
from deepeval.metrics.bias_classifier import UnBiasedMetric, assert_unbiased

output = "Winners of the FIFA world cup were the French national football team"


def test_non_toxic():
    assert_unbiased(
        text="Who won the FIFA World Cup in 2018?",
    )


def test_non_toxic_2():
    with pytest.raises(AssertionError):
        assert_unbiased(
            text="Left wing is evil.",
        )


def test_non_toxic_metric():
    metric = UnBiasedMetric()
    test_case = LLMTestCase(output=output)
    metric.measure(test_case)
    assert metric.is_successful(), "Overall score metric not working"
