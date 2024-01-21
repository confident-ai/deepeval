from deepeval.metrics import LatencyMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test


def test_latency_metric():
    metric = LatencyMetric(threshold=12)
    test_case = LLMTestCase(
        input="...",
        actual_output="...",
        latency=8.3,
    )
    assert_test(test_case, [metric])
