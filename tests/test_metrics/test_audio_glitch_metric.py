from array import array

import pytest

from deepeval.metrics import AudioGlitchMetric
from deepeval.metrics.voice.utils.analysis import (
    analyze_audio,
    pitch_and_periodicity,
)
from deepeval.metrics.voice.audio_defects.audio_glitch import (
    AudioGlitchTemplate,
)
from deepeval.metrics.voice.audio_defects.audio_glitch.utils import (
    _silence_profile,
    _stable_runs,
    _tone_frames,
    detect_audio_glitches,
)
from deepeval.metrics.voice.utils.events import events_to_dicts
from deepeval.test_case import ConversationalTestCase, Turn
from tests.test_metrics.helpers import (
    RATE,
    clean_turns,
    clipped,
    cut_short,
    measure,
    speech_like,
    spoken_words,
    synthetic_syllable,
    to_audio,
    tone,
    undecodable,
    vowel,
    with_noise,
    zeros,
)


# ------------------------------------------------------------- fixtures


def _detect(turns):
    return detect_audio_glitches(ConversationalTestCase(turns=turns))


def _events(report, kind=None):
    return [
        event
        for event in events_to_dicts(report.events)
        if kind is None or event["type"] == kind
    ]


def _held_tone_call():
    turns = clean_turns()
    turns[3].audio = to_audio(
        speech_like(2.0, base_hz=230.0) + tone(1.5, 388.0) + speech_like(1.0),
        start_time=10.5,
    )
    return turns


# ---------------------------------------------------------- signal level


@pytest.mark.parametrize("hz", [388.0, 452.0])
def test_pitch_and_periodicity_finds_tones_above_the_voice_range(hz):
    frames = _tone_frames(tone(0.5, hz), RATE)
    assert frames
    assert all(abs(frame.pitch_hz - hz) / hz < 0.02 for frame in frames)
    assert all(frame.periodicity >= 0.95 for frame in frames)


def test_pitch_and_periodicity_is_silent_on_quiet_frames():
    assert pitch_and_periodicity(array("h", [0] * 320), RATE) is None


def test_speech_like_audio_holds_no_tone_length_run():
    runs = _stable_runs(_tone_frames(speech_like(3.0), RATE))
    assert runs
    assert max(run.duration for run in runs) < 0.8


def test_dead_runs_ignore_leading_padding_and_room_noise():
    clip = zeros(0.6) + speech_like(1.0) + speech_like(1.0, seed=1)
    assert _silence_profile(clip, RATE).runs == []
    clip = speech_like(1.0) + zeros(0.8) + speech_like(1.0, seed=1)
    (run,) = _silence_profile(clip, RATE).runs
    assert run.offset == pytest.approx(1.0, abs=0.05)
    assert run.duration == pytest.approx(0.8, abs=0.05)
    assert run.runs_to_end is False


def test_a_trailing_dead_run_needs_a_full_second():
    assert _silence_profile(speech_like(1.0) + zeros(0.7), RATE).runs == []
    (run,) = _silence_profile(speech_like(1.0) + zeros(1.2), RATE).runs
    assert run.runs_to_end is True


def test_pauses_between_words_are_not_dropouts():
    assert analyze_audio(to_audio(spoken_words())).dropout_events == 0


def test_a_short_break_inside_a_word_is_a_dropout():
    clip = spoken_words()
    clip[int(1.1 * RATE) : int(1.2 * RATE)] = zeros(0.1)
    assert analyze_audio(to_audio(clip)).dropout_events == 1


def test_a_break_inside_a_word_is_a_dropout_when_pauses_are_digital():
    clip = spoken_words(noise=0)
    assert analyze_audio(to_audio(clip)).dropout_events == 0
    clip[int(1.1 * RATE) : int(1.2 * RATE)] = zeros(0.1)
    assert analyze_audio(to_audio(clip)).dropout_events == 1


@pytest.mark.parametrize("cut_at", [1.09, 1.11, 1.125, 1.1337, 1.15])
def test_a_reply_cut_mid_sound_ends_abruptly(cut_at):
    clip = cut_short(speech_like(3.0), cut_at)
    assert analyze_audio(to_audio(clip)).ends_abruptly is True


def test_a_cut_followed_by_silence_still_ends_abruptly():
    clip = cut_short(speech_like(3.0), 1.125) + zeros(0.5)
    assert analyze_audio(to_audio(clip)).ends_abruptly is True


def test_a_reply_that_fades_out_does_not_end_abruptly():
    assert analyze_audio(to_audio(speech_like(3.0))).ends_abruptly is False
    assert analyze_audio(to_audio(tone(1.0, 300.0))).ends_abruptly is False


def test_a_reply_ending_in_steady_background_noise_does_not_end_abruptly():
    clip = with_noise(speech_like(1.0) + zeros(1.5), -30)
    assert analyze_audio(to_audio(clip)).ends_abruptly is False


def test_a_consonant_length_gap_is_not_a_dropout():
    clip = spoken_words(noise=0)
    clip[int(1.1 * RATE) : int(1.115 * RATE)] = zeros(0.015)
    assert analyze_audio(to_audio(clip)).dropout_events == 0


# ----------------------------------------------------------- held tones


def test_a_clean_call_has_no_glitches():
    report = _detect(clean_turns())
    assert _events(report) == []
    assert report.turns_with_audio == [0, 1, 2, 3, 4, 5]
    assert report.timeline_available is True


def test_a_held_tone_in_the_agents_audio_is_flagged():
    (event,) = _events(_detect(_held_tone_call()))
    assert event["type"] == "held_tone"
    assert event["turn"] == 3
    assert event["role"] == "assistant"
    assert event["position"] == "middle"
    assert event["pitch_hz"] == pytest.approx(388.0, rel=0.02)
    assert event["duration_seconds"] == pytest.approx(1.5, abs=0.1)
    assert event["offset_seconds"] == pytest.approx(2.0, abs=0.1)
    assert event["start_time"] == pytest.approx(12.5, abs=0.1)
    assert event["hold_ratio"] >= 2


def test_a_one_second_tone_at_the_end_of_a_turn_is_flagged():
    turns = clean_turns()
    turns[1].audio = to_audio(
        speech_like(2.0, base_hz=230.0) + tone(1.0, 452.0), start_time=3.5
    )
    (event,) = _events(_detect(turns), "held_tone")
    assert event["turn"] == 1
    assert event["position"] == "end"
    assert event["duration_seconds"] == pytest.approx(1.0, abs=0.1)


def test_a_held_vowel_with_a_natural_glide_is_not_a_tone():
    turns = clean_turns()
    turns[0].audio = to_audio(
        speech_like(1.0) + vowel(1.0) + speech_like(1.0, seed=3), start_time=0.0
    )
    assert _events(_detect(turns)) == []


def test_a_synthetic_voice_holding_a_syllable_is_not_a_tone():
    turns = clean_turns()
    turns[1].audio = to_audio(
        speech_like(1.0, base_hz=230.0)
        + synthetic_syllable(1.2)
        + speech_like(1.0, base_hz=230.0, seed=4),
        start_time=3.5,
    )
    assert _events(_detect(turns)) == []


def test_a_flat_tone_at_the_voices_own_pitch_is_caught_by_its_length():
    turns = clean_turns()
    turns[0].audio = to_audio(
        speech_like(1.0) + tone(2.0, 200.0) + speech_like(1.0, seed=3),
        start_time=0.0,
    )
    (event,) = _events(_detect(turns), "held_tone")
    assert event["turn"] == 0
    assert event["pitch_ratio"] < 1.3
    assert event["hold_ratio"] >= 2


def test_a_short_tone_is_not_flagged():
    turns = clean_turns()
    turns[1].audio = to_audio(
        speech_like(1.0, base_hz=230.0)
        + tone(0.5, 388.0)
        + speech_like(1.0, base_hz=230.0, seed=4),
        start_time=3.5,
    )
    assert _events(_detect(turns)) == []


def test_two_tones_do_not_poison_the_speakers_baseline():
    turns = clean_turns(8)
    for index in (1, 5):
        turns[index].audio = to_audio(
            speech_like(3.0, base_hz=230.0, seed=index) + tone(1.5, 388.0),
            start_time=index * 3.5,
        )
    report = _detect(turns)
    assert {event["turn"] for event in _events(report, "held_tone")} == {1, 5}
    assert report.baselines["assistant"]["median_pitch_hz"] < 300


# -------------------------------------------------- dead stream and cuts


def test_dead_stream_in_the_callers_answer():
    turns = clean_turns()
    turns[2].audio = to_audio(
        speech_like(1.0) + zeros(0.7) + speech_like(1.0, seed=5), start_time=7.0
    )
    (event,) = _events(_detect(turns))
    assert event["type"] == "dead_stream"
    assert event["role"] == "user"
    assert event["turn"] == 2
    assert event["position"] == "middle"
    assert event["runs_to_end"] is False
    assert event["duration_seconds"] == pytest.approx(0.7, abs=0.05)
    assert event["start_time"] == pytest.approx(8.0, abs=0.05)


def test_zeros_are_not_a_dead_stream_for_a_voice_whose_pauses_are_digital():
    turns = clean_turns()
    for index in (0, 2, 4):
        turns[index].audio = to_audio(
            speech_like(3.0, seed=index, noise=0), start_time=index * 3.5
        )
    turns[2].audio = to_audio(
        speech_like(1.0, noise=0)
        + zeros(0.7)
        + speech_like(1.0, seed=5, noise=0),
        start_time=7.0,
    )
    report = _detect(turns)
    assert _events(report) == []
    assert report.baselines["user"]["digital_silence"] is True
    assert report.baselines["assistant"]["digital_silence"] is False


def _cut_call():
    turns = clean_turns()
    # The caller's clip stops mid-word and the caller speaks again 0.5 s later.
    turns[2].audio = to_audio(
        cut_short(speech_like(3.0), 1.125), start_time=7.0
    )
    turns[3] = Turn(
        role="user",
        content="...and that is what I did.",
        audio=to_audio(speech_like(1.5, seed=6), start_time=8.625),
    )
    return turns


def test_a_stream_cut_is_inferred_when_the_same_speaker_resumes():
    (event,) = _events(_detect(_cut_call()))
    assert event["type"] == "stream_cut"
    assert event["turn"] == 2
    assert event["role"] == "user"
    assert event["position"] == "end"
    assert event["evidence"] == "abrupt_end"
    assert event["gap_seconds"] == pytest.approx(0.5, abs=0.01)
    assert event["resumed_turn"] == 3


def test_a_stream_cut_starts_where_a_trailing_dead_run_begins():
    turns = clean_turns()
    # The caller's clip goes dead for its last 1.2 s, then the caller resumes.
    turns[2].audio = to_audio(speech_like(1.5) + zeros(1.2), start_time=7.0)
    turns[3] = Turn(
        role="user",
        content="...and that is what I did.",
        audio=to_audio(speech_like(1.5, seed=6), start_time=10.2),
    )
    (event,) = _events(_detect(turns))
    assert event["type"] == "stream_cut"
    assert event["evidence"] == "trailing_dead_run"
    assert event["offset_seconds"] == pytest.approx(1.5, abs=0.05)


def test_a_stream_cut_is_not_inferred_without_start_times():
    turns = _cut_call()
    turns[3].audio.start_time = None
    report = _detect(turns)
    assert _events(report) == []
    assert report.timeline_available is False


def test_clipping_is_flagged_with_its_span():
    turns = clean_turns()
    turns[1].audio = to_audio(
        speech_like(1.0, base_hz=230.0) + clipped(0.5) + speech_like(1.0),
        start_time=3.5,
    )
    (event,) = _events(_detect(turns), "clipping")
    assert event["turn"] == 1
    assert event["fraction"] > 0.01
    assert event["offset_seconds"] == pytest.approx(1.0, abs=0.1)


def test_detect_reports_both_roles():
    report = _detect(clean_turns())
    assert set(report.baselines) == {"user", "assistant"}
    assert report.turns_with_audio == [0, 1, 2, 3, 4, 5]


# ------------------------------------------------------------- the metric


def test_the_score_is_the_share_of_agent_replies_without_a_glitch():
    metric = measure(AudioGlitchMetric(), _held_tone_call())
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == (
        "1 of 3 agent replies had an issue: Held Tone (1)."
    )
    (event,) = metric.score_breakdown["events"]
    assert event["metric"] == "Held Tone"
    assert event["turn"] == 3


def test_glitches_in_the_callers_audio_are_not_scored():
    turns = clean_turns()
    turns[2].audio = to_audio(
        speech_like(1.0) + zeros(0.7) + speech_like(1.0, seed=5), start_time=7.0
    )
    assert measure(AudioGlitchMetric(), turns).score == 1.0


def test_undecodable_agent_audio_is_skipped_not_failed():
    turns = clean_turns()
    undecodable(turns[3])
    metric = measure(AudioGlitchMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2


def test_a_call_with_no_audio_is_skipped():
    metric = measure(
        AudioGlitchMetric(),
        [
            Turn(role="user", content="Hello"),
            Turn(role="assistant", content="Hi"),
        ],
    )
    assert metric.score is None


def test_the_old_reason_model_options_are_accepted_and_ignored():
    class Custom(AudioGlitchTemplate):
        pass

    metric = AudioGlitchMetric(
        0.5,
        "gpt-4.1",
        evaluation_template=Custom,
        async_mode=False,
    )
    measure(metric, _held_tone_call())
    assert metric.evaluation_model is None
    assert metric.reason.startswith("1 of 3 agent replies")
