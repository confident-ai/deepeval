"""Test BERT score
"""
import pytest
from deepeval.metrics.bertscore_metric import BertScoreMetric


def test_bert_score():
    scorer = BertScoreMetric()
    score = scorer.measure("Why are you weird", "Why are you strange?")
    assert scorer.is_successful()
