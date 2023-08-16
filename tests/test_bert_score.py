"""Test BERT score
"""

from deepeval.metrics.bertscore import BertScore


def test_bert_score():
    scorer = BertScore()
    score = scorer.measure("Why are you weird", "Why are you strange?")
    assert scorer.is_successful()
