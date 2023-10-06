"""Tests for answer length
"""
from deepeval.test_case import LLMTestCase
from deepeval.metrics.answer_length import LengthMetric
from deepeval.run_test import assert_test


def test_answer_length():
    metric = LengthMetric()
    test_case = LLMTestCase(
        query="placeholder",
        output=" Some output ",
        expected_output="Some output",
    )
    assert_test(test_case, [metric])
