from deepeval.metrics import VoiceConsistencyMetric
from tests.test_metrics.helpers import clean_turns, measure


def test_fewer_than_three_replies_cannot_be_compared():
    metric = measure(VoiceConsistencyMetric(), clean_turns(4))
    assert metric.score is None
    assert metric.skipped is True
    assert {
        part["score"] for part in metric.score_breakdown["metrics"].values()
    } == {None}
