import pytest

from deepeval.metrics import CostMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test


def test_cost():
    test_case = LLMTestCase(input="...", actual_output="...", cost=0.4)
    assert_test(test_case, [CostMetric(max_cost=5)])
