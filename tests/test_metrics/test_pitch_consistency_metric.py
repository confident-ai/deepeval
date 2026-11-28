import pytest

from deepeval.metrics import PitchConsistencyMetric
from deepeval.metrics.voice.utils.analysis import analyze_audio, semitones
from tests.test_metrics.helpers import (
    clean_turns,
    edit_reply,
    measure,
    set_audio,
    speech_like,
    to_audio,
)


def _call(count=6):
    """Replies long enough to read a base pitch from."""
    return clean_turns(count, seconds=6.0, rhythm=2.5)


def _reply(semitones_up=0.0, *, seconds=6.0, swing=1.0):
    """An agent reply moved `semitones_up`, its swings scaled by `swing`."""
    ratio = 2 ** (semitones_up / 12)
    return speech_like(
        seconds,
        base_hz=230.0 * ratio,
        glide_hz=30.0 * ratio * swing,
        seed=1,
        rhythm=2.5,
    )


def _shifted(semitones_up):
    return edit_reply(1, set_audio(_reply(semitones_up)), _call())


@pytest.mark.parametrize("semitones_up,higher", [(4, True), (-4, False)])
def test_a_reply_four_semitones_off_fails_pitch_consistency(
    semitones_up, higher
):
    metric = measure(PitchConsistencyMetric(), _shifted(semitones_up))
    assert metric.score == pytest.approx(2 / 3)
    [event] = metric.score_breakdown["events"]
    assert event["type"] == "pitch_shift"
    assert event["turn"] == 1
    assert event["shift_semitones"] == pytest.approx(semitones_up, abs=0.1)
    assert event["description"] == (
        f"Voice sat 4 semitones {'higher' if higher else 'lower'} than in "
        "the agent's other replies"
    )


@pytest.mark.parametrize("semitones_up", [2, -2, 2.9])
def test_a_reply_under_three_semitones_off_is_not_a_pitch_shift(semitones_up):
    metric = measure(PitchConsistencyMetric(), _shifted(semitones_up))
    assert metric.score == 1.0
    assert metric.score_breakdown["events"] == []


def test_pitch_consistency_limit_is_configurable():
    turns = _shifted(6)
    assert (
        measure(PitchConsistencyMetric(max_shift_semitones=7), turns).score
        == 1.0
    )
    assert measure(
        PitchConsistencyMetric(max_shift_semitones=1), _shifted(2)
    ).score == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        PitchConsistencyMetric(max_shift_semitones=0)


def test_pitch_events_hold_plain_numbers():
    [event] = measure(PitchConsistencyMetric(), _shifted(6)).score_breakdown[
        "events"
    ]
    assert type(event["pitch_base_hz"]) is float
    assert "pitch_median_hz" not in event


def test_a_reply_with_under_three_seconds_of_voiced_speech_is_skipped():
    turns = edit_reply(1, set_audio(_reply(6, seconds=3.0)), _call(8))
    metric = measure(PitchConsistencyMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["skipped"] == [
        {
            "turn": 1,
            "error": "Reply has under 3s of voiced speech to follow its "
            "pitch.",
        }
    ]


def test_wider_upward_swings_from_the_same_base_pitch_are_not_a_pitch_shift():
    ordinary = analyze_audio(to_audio(_reply()))
    swinging = analyze_audio(to_audio(_reply(4.6, swing=2.5)))
    assert semitones(swinging.pitch_median_hz, ordinary.pitch_median_hz) > 4
    assert swinging.pitch_base_hz == pytest.approx(
        ordinary.pitch_base_hz, rel=0.02
    )
    turns = edit_reply(1, set_audio(_reply(4.6, swing=2.5)), _call())
    assert measure(PitchConsistencyMetric(), turns).score == 1.0


def test_the_base_pitch_is_the_low_end_of_a_replys_voiced_pitch():
    reply = analyze_audio(to_audio(_reply()))
    assert reply.pitch_base_hz == pytest.approx(200.0, rel=0.02)
    assert reply.pitch_base_hz < reply.pitch_median_hz
    assert len(reply.voiced_pitches_hz) == reply.voiced_pitch_frames


def test_a_few_very_low_frames_do_not_shift_the_base_pitch():
    # 0.9 s of a 6 s reply spoken 8 semitones down, as creaky frames read.
    creaky = _reply(seconds=5.1) + speech_like(
        0.9, base_hz=230.0 * 2 ** (-8 / 12), glide_hz=5.0, seed=1, rhythm=2.5
    )
    ordinary = analyze_audio(to_audio(_reply()))
    dipped = analyze_audio(to_audio(creaky))
    assert semitones(dipped.pitch_base_hz, ordinary.pitch_base_hz) < -3
    assert (
        abs(semitones(dipped.pitch_base_check_hz, ordinary.pitch_base_check_hz))
        < 2
    )
    turns = edit_reply(1, set_audio(creaky), _call())
    assert measure(PitchConsistencyMetric(), turns).score == 1.0
