"""Tests for toxic
"""


import pytest
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import UnBiasedMetric
from deepeval.evaluator import assert_test

output = "Winners of the FIFA world cup were the French national football team"

metric = UnBiasedMetric(evaluation_params=[LLMTestCaseParams.INPUT])


def test_non_bias_metric():
    metric = UnBiasedMetric(evaluation_params=[LLMTestCaseParams.INPUT])
    test_case = LLMTestCase(input="placeholder", actual_output=output)
    assert_test(test_case, [metric])
