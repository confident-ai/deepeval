from deepeval.metrics import CostMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test


def test_cost_metric():
    metric = CostMetric(threshold=12)
    test_case = LLMTestCase(input="...", actual_output="...", cost=12)
    assert_test(test_case, [metric])
