import pytest

from deepeval.metrics import LoudnessMetric
from tests.test_metrics.helpers import (
    agent_speech,
    measure,
    with_gain,
    with_reply_audio,
)


def test_loudness_limits_are_configurable():
    loud = with_reply_audio(1, with_gain(agent_speech(), 3))
    assert measure(LoudnessMetric(), loud).score == 1.0
    strict = measure(LoudnessMetric(min_dbfs=-40, max_dbfs=-12), loud)
    assert [event["type"] for event in strict.score_breakdown["events"]] == [
        "too_loud"
    ]
    assert (
        strict.reason
        == "1 of 3 agent replies had speech outside -40 to -12 dBFS."
    )
    quiet = with_reply_audio(1, with_gain(agent_speech(), 0.05))
    assert measure(LoudnessMetric(min_dbfs=-60), quiet).score == 1.0


@pytest.mark.parametrize("min_dbfs,max_dbfs", [(-6, -6), (-6, -32), (-32, 3)])
def test_invalid_loudness_limits_are_rejected(min_dbfs, max_dbfs):
    with pytest.raises(ValueError):
        LoudnessMetric(min_dbfs=min_dbfs, max_dbfs=max_dbfs)
