import pytest

from deepeval.metrics import VolumeVarianceMetric
from tests.test_metrics.helpers import (
    agent_speech,
    clean_turns,
    faded,
    measure,
    to_audio,
    with_reply_audio,
)


@pytest.mark.parametrize("db", [-20, 20])
def test_a_fade_or_swell_is_a_volume_drift(db):
    metric = measure(
        VolumeVarianceMetric(),
        with_reply_audio(1, faded(agent_speech(6.0), db)),
    )
    [event] = metric.score_breakdown["events"]
    assert event["drift_db"] * db > 0
    assert abs(event["drift_db"]) > 10


def test_steady_speech_has_no_volume_drift():
    metric = measure(
        VolumeVarianceMetric(), with_reply_audio(1, agent_speech(6.0))
    )
    assert metric.score == 1.0


def test_volume_drift_limit_is_configurable():
    turns = with_reply_audio(1, faded(agent_speech(6.0), -20))
    assert measure(VolumeVarianceMetric(max_drift_db=20), turns).score == 1.0
    with pytest.raises(ValueError):
        VolumeVarianceMetric(max_drift_db=0)


def test_a_reply_too_short_to_follow_is_skipped():
    metric = measure(
        VolumeVarianceMetric(), with_reply_audio(1, agent_speech(1.5))
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]


def test_a_call_of_only_short_replies_is_skipped():
    turns = clean_turns()
    for index in (1, 3, 5):
        turns[index].audio = to_audio(agent_speech(1.5))
    metric = measure(VolumeVarianceMetric(), turns)
    assert metric.score is None
    assert metric.reason == (
        "No assistant reply was long enough to follow its volume."
    )
