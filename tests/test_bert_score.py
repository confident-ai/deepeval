"""Test BERT score
"""
from deepeval.metrics.bertscore_metric import BertScoreMetric
from deepeval.test_case import LLMTestCase


def test_bert_score():
    metric = BertScoreMetric()
    test_case = LLMTestCase(
        output="Why are you weird", expected_output="Why are you strange?"
    )
    score = metric.measure(test_case)
    assert metric.is_successful(), f"failed - {score}"
