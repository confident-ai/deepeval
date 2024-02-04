import pytest
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import NERMetric
from deepeval import assert_test


def test_ner_metric():
    """Test Chat Completion"""
    metric = NERMetric(minimum_score=0.5)
    actual_output_str = " ".join(["B-PER", "I-PER", "B-LOC"])
    test_case = LLMTestCase(
        input="George Bush visited China in 2006.",
        actual_output=actual_output_str
    )
    assert_test(test_case, [metric])
