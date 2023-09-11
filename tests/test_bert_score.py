"""Test BERT score
"""
from deepeval.metrics.bertscore_metric import BertScoreMetric
from deepeval.test_case import LLMTestCase
from deepeval.run_test import assert_test


def test_bert_score():
    metric = BertScoreMetric()
    test_case = LLMTestCase(
        output="Why are you weird", expected_output="Why are you strange?"
    )
    assert_test(test_case, [metric])
