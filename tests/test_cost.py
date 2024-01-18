from deepeval.metrics import CostMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test


def test_cost_metric():
    metric = CostMetric(threshold=12)
    test_case = LLMTestCase(input="...", actual_output="...", cost=123)
    assert_test(test_case, [metric])
