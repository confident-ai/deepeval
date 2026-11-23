import pytest

from deepeval.metrics import PausesMetric
from tests.test_metrics.helpers import (
    agent_speech,
    edit_reply,
    measure,
    set_audio,
    with_noise,
    with_pause,
)


def test_silence_before_the_reply_is_not_a_pause():
    turns = edit_reply(1, set_audio(with_pause(agent_speech(), 0.0, 5.0)))
    assert measure(PausesMetric(), turns).score == 1.0


def test_a_pause_over_background_noise_is_still_a_pause():
    noisy = with_noise(with_pause(agent_speech(), 1.5, 3.0), -40)
    turns = edit_reply(1, set_audio(noisy))
    [event] = measure(PausesMetric(), turns).score_breakdown["events"]
    assert event["type"] == "long_pause"


def test_too_much_silence_fails_pauses():
    gappy = agent_speech()
    for at in (0.6, 1.4, 2.2):
        gappy = with_pause(gappy, at, 0.6)
    turns = edit_reply(1, set_audio(gappy))
    [event] = measure(PausesMetric(), turns).score_breakdown["events"]
    assert event["type"] == "too_much_silence"


def test_pause_limits_are_configurable():
    turns = edit_reply(1, set_audio(with_pause(agent_speech(), 1.5, 3.0)))
    assert (
        measure(
            PausesMetric(max_pause_ms=4000, max_pause_percent=90), turns
        ).score
        == 1.0
    )
    with pytest.raises(ValueError):
        PausesMetric(max_pause_ms=0)
    with pytest.raises(ValueError):
        PausesMetric(max_pause_percent=101)


def test_a_reply_too_short_for_pauses_is_skipped():
    turns = edit_reply(1, set_audio(agent_speech(1.5)))
    metric = measure(PausesMetric(), turns)
    assert metric.score == 1.0
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]
