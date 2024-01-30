from deepeval.metrics import LatencyMetric, CostMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test


def test_cost_metric():
    metric = CostMetric(max_cost=12)
    test_case = LLMTestCase(input="...", actual_output="...", cost=12)
    assert_test(test_case, [metric])


def test_latency_metric():
    metric = LatencyMetric(max_latency=12)
    test_case = LLMTestCase(
        input="...",
        actual_output="...",
        latency=8.3,
    )
    assert_test(test_case, [metric])
