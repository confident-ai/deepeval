import pytest

from deepeval.metrics import AbruptPitchJumpMetric
from deepeval.metrics.voice.utils.analysis import (
    pitch_jump_frames,
    pitch_track,
)
from tests.test_metrics.system_one_fakes import FakeAudioDecisionModel
from tests.test_metrics.helpers import (
    clean_turns,
    RATE,
    cracked_vowel,
    edit_reply,
    measure,
    set_audio,
    vowel,
)


@pytest.mark.parametrize("semitones", [8, 12])
def test_a_voice_crack_is_a_pitch_jump(semitones):
    turns = edit_reply(1, set_audio(cracked_vowel(3.0, semitones=semitones)))
    metric = measure(AbruptPitchJumpMetric(), turns)
    assert metric.score == pytest.approx(2 / 3)


def test_a_jump_up_out_of_the_voices_range_is_a_pitch_jump():
    turns = edit_reply(1, set_audio(cracked_vowel(3.0, semitones=8)))
    [event] = measure(AbruptPitchJumpMetric(), turns).score_breakdown["events"]
    assert event["type"] == "pitch_jump"
    assert event["turn"] == 1
    assert event["pitch_hz"] == pytest.approx(_raised(8), rel=0.02)
    assert type(event["pitch_hz"]) is float
    assert event["offset_seconds"] == pytest.approx(0.96)
    assert event["description"] == (
        "Pitch jumped abruptly to 4 semitones above the voice's usual range"
    )


@pytest.mark.parametrize("semitones", [-8, -12])
def test_a_jump_down_is_not_a_pitch_jump(semitones):
    turns = edit_reply(1, set_audio(cracked_vowel(3.0, semitones=semitones)))
    metric = measure(AbruptPitchJumpMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3


@pytest.mark.parametrize(
    "semitones_under_landing,score", [(0.5, 1.0), (1.5, pytest.approx(2 / 3))]
)
def test_a_jump_must_land_a_semitone_above_pitch_the_voice_uses_elsewhere(
    semitones_under_landing, score
):
    elsewhere_hz = _raised(8 - semitones_under_landing)
    turns = edit_reply(1, set_audio(cracked_vowel(3.0, semitones=8)))
    edit_reply(
        3,
        set_audio(vowel(3.0, start_hz=elsewhere_hz, end_hz=elsewhere_hz)),
        turns,
    )
    assert measure(AbruptPitchJumpMetric(), turns).score == score


@pytest.mark.parametrize(
    "crack_seconds,score", [(0.32, pytest.approx(2 / 3)), (0.4, 1.0)]
)
def test_a_jump_to_pitch_held_for_over_5_percent_of_the_call_is_in_range(
    crack_seconds, score
):
    turns = edit_reply(
        1, set_audio(cracked_vowel(3.0, crack_seconds=crack_seconds))
    )
    assert measure(AbruptPitchJumpMetric(), turns).score == score


def test_a_steady_vowel_has_no_pitch_jump():
    turns = edit_reply(1, set_audio(cracked_vowel(3.0, semitones=0)))
    assert measure(AbruptPitchJumpMetric(), turns).score == 1.0


def test_a_step_held_under_200_ms_is_not_a_pitch_jump():
    turns = edit_reply(1, set_audio(cracked_vowel(3.0, crack_seconds=0.16)))
    assert measure(AbruptPitchJumpMetric(), turns).score == 1.0


@pytest.mark.parametrize(
    "held_seconds,score", [(0.16, 1.0), (0.2, pytest.approx(2 / 3))]
)
def test_a_step_at_the_end_of_a_reply_needs_200_ms_after_it(
    held_seconds, score
):
    stepped = cracked_vowel(
        3.0, crack_at=3.0 - held_seconds, crack_seconds=held_seconds
    )
    turns = edit_reply(1, set_audio(stepped))
    assert measure(AbruptPitchJumpMetric(), turns).score == score


def _raised(semitones, hz=200.0):
    return hz * 2 ** (semitones / 12)


@pytest.mark.parametrize(
    "track,jumps",
    [
        ([200.0] * 10 + [_raised(8)] * 5, [10]),
        ([200.0] * 10 + [_raised(8)] * 4, []),
        ([200.0] * 4 + [_raised(8)] * 10, []),
    ],
)
def test_a_pitch_jump_needs_five_steady_frames_on_both_sides(track, jumps):
    assert pitch_jump_frames(track) == jumps


def test_a_pitch_jump_is_measured_between_the_steady_stretches():
    one_high_frame = [200.0] * 5 + [_raised(7)] + [_raised(5)] * 4
    one_low_frame = [200.0] * 5 + [_raised(5)] + [_raised(7)] * 4
    assert pitch_jump_frames(one_high_frame) == []
    assert pitch_jump_frames(one_low_frame) == [5]


@pytest.mark.parametrize("hz", [80.0, 115.0, 200.0, 300.0, 480.0])
def test_a_steady_vowel_is_tracked_at_its_pitch(hz):
    track = pitch_track(vowel(1.5, start_hz=hz, end_hz=hz), RATE)
    assert len(track) > 30
    assert all(pitch == pytest.approx(hz, rel=0.02) for pitch in track)


def test_an_octave_up_crack_is_tracked_an_octave_up():
    track = pitch_track(cracked_vowel(3.0, semitones=12), RATE)
    assert all(
        pitch == pytest.approx(400.0, rel=0.02) for pitch in track[25:30]
    )
    assert all(
        pitch == pytest.approx(200.0, rel=0.02)
        for pitch in track[:25] + track[30:]
    )
    assert pitch_jump_frames(track) == [25, 30]


def _with_model(samples, probability):
    model = FakeAudioDecisionModel(probability=probability)
    metric = measure(
        AbruptPitchJumpMetric(eval_mode="hybrid", audio_decision_model=model),
        edit_reply(1, set_audio(samples)),
    )
    return metric, model


def test_a_short_step_up_is_only_flagged_with_a_decision_model():
    stepped = cracked_vowel(3.0, crack_seconds=0.12)
    assert (
        measure(
            AbruptPitchJumpMetric(), edit_reply(1, set_audio(stepped))
        ).score
        == 1.0
    )
    metric, model = _with_model(stepped, 0.9)
    assert metric.score == pytest.approx(2 / 3)
    assert len(model.clips) == 1


def test_a_step_the_decision_model_finds_natural_is_dismissed():
    metric, _ = _with_model(cracked_vowel(3.0, crack_seconds=0.12), 0.1)
    assert metric.score == 1.0
    assert metric.score_breakdown["dismissed_events"] == 1


def test_a_step_down_is_never_sent_to_the_decision_model():
    metric, model = _with_model(cracked_vowel(3.0, semitones=-8), 0.9)
    assert metric.score == 1.0
    assert model.clips == []


def test_a_step_landing_inside_the_voices_range_is_not_sent():
    low = cracked_vowel(3.0, semitones=0)
    high = cracked_vowel(3.0, hz=200.0 * 2 ** (8 / 12), semitones=0)
    turns = clean_turns()
    set_audio(cracked_vowel(3.0, crack_seconds=0.12))(turns[1])
    set_audio(high)(turns[3])
    set_audio(low)(turns[5])
    model = FakeAudioDecisionModel(probability=0.9)
    metric = measure(
        AbruptPitchJumpMetric(eval_mode="hybrid", audio_decision_model=model),
        turns,
    )
    assert metric.score == 1.0
    assert model.clips == []
