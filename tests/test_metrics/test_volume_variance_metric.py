import pytest

from deepeval.metrics import VolumeVarianceMetric
from tests.test_metrics.helpers import (
    agent_speech,
    clean_turns,
    faded,
    measure,
    to_audio,
    with_quiet_tail,
    with_reply_audio,
)


@pytest.mark.parametrize("db", [-22, 22])
def test_a_fade_or_swell_is_a_volume_drift(db):
    metric = measure(
        VolumeVarianceMetric(),
        with_reply_audio(1, faded(agent_speech(8.0), db)),
    )
    [event] = metric.score_breakdown["events"]
    assert event["drift_db"] * db > 0
    assert abs(event["drift_db"]) > 15


def test_a_drift_of_ten_decibels_is_not_flagged():
    metric = measure(
        VolumeVarianceMetric(),
        with_reply_audio(
            1, faded(agent_speech(8.0), -11), clean_turns(seconds=6.0)
        ),
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3


def test_a_reply_that_only_drops_in_its_final_second_is_not_a_drift():
    metric = measure(
        VolumeVarianceMetric(max_drift_db=5),
        with_reply_audio(1, with_quiet_tail(agent_speech(8.0), -18)),
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 1
    assert metric.score_breakdown["events"] == []


def test_a_volume_that_moves_both_ways_is_not_a_drift():
    reply = faded(agent_speech(4.0), -18) + faded(agent_speech(4.0), 18)
    metric = measure(
        VolumeVarianceMetric(max_drift_db=5), with_reply_audio(1, reply)
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["events"] == []


def test_steady_speech_has_no_volume_drift():
    metric = measure(
        VolumeVarianceMetric(), with_reply_audio(1, agent_speech(6.0))
    )
    assert metric.score == 1.0


def test_volume_drift_limit_is_configurable():
    turns = with_reply_audio(1, faded(agent_speech(8.0), -20))
    assert measure(VolumeVarianceMetric(max_drift_db=20), turns).score == 1.0
    with pytest.raises(ValueError):
        VolumeVarianceMetric(max_drift_db=0)


def test_a_reply_under_six_seconds_is_too_short_to_follow():
    metric = measure(
        VolumeVarianceMetric(),
        with_reply_audio(1, agent_speech(5.0), clean_turns(seconds=6.0)),
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert metric.score_breakdown["skipped"] == [
        {"turn": 1, "error": "Reply has under 6 seconds of speech to follow."}
    ]


def test_a_call_of_only_short_replies_is_skipped():
    turns = clean_turns()
    for index in (1, 3, 5):
        turns[index].audio = to_audio(agent_speech(3.0))
    metric = measure(VolumeVarianceMetric(), turns)
    assert metric.score is None
    assert metric.reason == (
        "No assistant reply was long enough to follow its volume."
    )
