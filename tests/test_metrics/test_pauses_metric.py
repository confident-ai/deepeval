import pytest

from deepeval.metrics import PausesMetric
from deepeval.metrics.voice.prosody.pauses.pauses import EVENT_QUESTIONS
from tests.test_metrics.helpers import (
    agent_speech,
    edit_reply,
    measure,
    set_audio,
    with_noise,
    with_pause,
)
from tests.test_metrics.system_one_fakes import FakeAudioDecisionModel


def test_silence_before_the_reply_is_not_a_pause():
    turns = edit_reply(1, set_audio(with_pause(agent_speech(), 0.0, 5.0)))
    assert measure(PausesMetric(), turns).score == 1.0


def test_a_pause_over_background_noise_is_still_a_pause():
    noisy = with_noise(with_pause(agent_speech(), 1.5, 5.0), -40)
    turns = edit_reply(1, set_audio(noisy))
    [event] = measure(PausesMetric(), turns).score_breakdown["events"]
    assert event["type"] == "long_pause"


def test_a_pause_under_four_seconds_is_not_flagged():
    flowing = agent_speech(6.0, rhythm=2.5)
    turns = edit_reply(1, set_audio(with_pause(flowing, 1.5, 3.0)))
    metric = measure(PausesMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3


def test_a_pause_over_four_seconds_is_a_long_pause():
    flowing = agent_speech(6.0, rhythm=2.5)
    turns = edit_reply(1, set_audio(with_pause(flowing, 1.5, 5.0)))
    [event] = measure(PausesMetric(), turns).score_breakdown["events"]
    assert event["type"] == "long_pause"
    assert event["longest_pause_ms"] == pytest.approx(5000, abs=200)


def test_too_much_silence_fails_pauses():
    gappy = agent_speech()
    for at in (0.6, 1.4, 2.2):
        gappy = with_pause(gappy, at, 0.6)
    turns = edit_reply(1, set_audio(gappy))
    [event] = measure(PausesMetric(), turns).score_breakdown["events"]
    assert event["type"] == "too_much_silence"


def test_pause_limits_are_configurable():
    turns = edit_reply(1, set_audio(with_pause(agent_speech(), 1.5, 5.0)))
    assert (
        measure(
            PausesMetric(max_pause_ms=6000, max_pause_percent=90), turns
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


def _gappy_speech():
    gappy = agent_speech()
    for at in (0.6, 1.4, 2.2):
        gappy = with_pause(gappy, at, 0.6)
    return gappy


@pytest.mark.parametrize(
    "event_type,speech",
    [
        (
            "long_pause",
            lambda: with_pause(agent_speech(6.0, rhythm=2.5), 1.5, 5.0),
        ),
        ("too_much_silence", _gappy_speech),
    ],
)
def test_a_pause_the_decision_model_cannot_hear_is_dismissed(
    event_type, speech
):
    model = FakeAudioDecisionModel(probability=0.3)
    metric = measure(
        PausesMetric(eval_mode="hybrid", audio_decision_model=model),
        edit_reply(1, set_audio(speech())),
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["dismissed_events"] == 1
    assert model.questions == [EVENT_QUESTIONS[event_type]]
