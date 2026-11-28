import pytest

from deepeval.metrics import LoudnessMetric
from tests.test_metrics.helpers import (
    agent_speech,
    measure,
    with_gain,
    with_reply_audio,
)


def test_loudness_limits_are_configurable():
    loud = with_reply_audio(1, with_gain(agent_speech(), 2))
    assert measure(LoudnessMetric(), loud).score == 1.0
    strict = measure(LoudnessMetric(min_dbfs=-40, max_dbfs=-16), loud)
    assert [event["type"] for event in strict.score_breakdown["events"]] == [
        "too_loud"
    ]
    assert (
        strict.reason
        == "1 of 3 agent replies had speech outside -40 to -16 dBFS."
    )
    quiet = with_reply_audio(1, with_gain(agent_speech(), 0.05))
    assert measure(LoudnessMetric(min_dbfs=-60), quiet).score == 1.0


@pytest.mark.parametrize("min_dbfs,max_dbfs", [(-6, -6), (-6, -32), (-32, 3)])
def test_invalid_loudness_limits_are_rejected(min_dbfs, max_dbfs):
    with pytest.raises(ValueError):
        LoudnessMetric(min_dbfs=min_dbfs, max_dbfs=max_dbfs)


def test_a_reply_with_under_a_second_of_speech_is_skipped_for_loudness():
    quiet = with_gain(agent_speech(1.0), 0.15)
    metric = measure(LoudnessMetric(), with_reply_audio(1, quiet))
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert metric.score_breakdown["skipped"] == [
        {"turn": 1, "error": "Reply has under 1 second of speech."}
    ]


def test_a_quiet_reply_with_over_a_second_of_speech_is_too_quiet():
    quiet = with_gain(agent_speech(1.5), 0.15)
    metric = measure(LoudnessMetric(), with_reply_audio(1, quiet))
    assert [event["type"] for event in metric.score_breakdown["events"]] == [
        "too_quiet"
    ]


def test_a_reply_above_minus_ten_dbfs_is_too_loud():
    loud = with_reply_audio(1, with_gain(agent_speech(), 3))
    [event] = measure(LoudnessMetric(), loud).score_breakdown["events"]
    assert event["type"] == "too_loud"
    assert measure(LoudnessMetric(max_dbfs=-6), loud).score == 1.0
