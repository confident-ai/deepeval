"""Deterministic metrics never call a model, so an image in the test case
must not trip the multimodal-model check. No API key needed."""

from deepeval.metrics import ExactMatchMetric, PatternMatchMetric
from deepeval.metrics.agent_loop_detection import AgentLoopDetectionMetric
from deepeval.test_case import LLMTestCase, MLLMImage

IMAGE = MLLMImage(url="https://example.com/receipt.png")


def _test_case() -> LLMTestCase:
    test_case = LLMTestCase(
        input=f"What is the total on this receipt? {IMAGE}",
        actual_output="$42.00",
        expected_output="$42.00",
    )
    assert test_case.multimodal is True
    return test_case


def test_exact_match_scores_multimodal_test_case():
    metric = ExactMatchMetric()
    assert metric.measure(_test_case(), _show_indicator=False) == 1.0
    assert metric.success is True


def test_pattern_match_scores_multimodal_test_case():
    metric = PatternMatchMetric(pattern=r"\$\d+\.\d{2}")
    assert metric.measure(_test_case(), _show_indicator=False) == 1.0
    assert metric.success is True


def test_agent_loop_detection_scores_multimodal_test_case():
    test_case = _test_case()
    test_case._trace_dict = {
        "type": "agent",
        "name": "agent",
        "input": "query",
        "output": "answer",
        "children": [],
    }
    metric = AgentLoopDetectionMetric(async_mode=False)
    assert metric.measure(test_case, _show_indicator=False) == 1.0
    assert metric.success is True
