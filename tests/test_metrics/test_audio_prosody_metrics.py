import pytest

from deepeval.metrics import (
    AbruptPitchJumpMetric,
    LoudnessConsistencyMetric,
    LoudnessMetric,
    PausesMetric,
    PitchConsistencyMetric,
    PitchVariationMetric,
    SpeakingRateMetric,
)
from deepeval.metrics.voice.prosody.speaking_rate.utils import (
    speech_span_rate_wpm,
)
from deepeval.metrics.voice.utils.analysis import analyze_audio
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from tests.test_metrics.helpers import (
    clean_turns,
    cracked_vowel,
    speech_like,
    to_audio,
    with_gain,
    with_noise,
    with_pause,
    zeros,
)


def agent_speech(seconds=3.0, **kwargs):
    return speech_like(seconds, base_hz=230.0, seed=1, **kwargs)


def _set_audio(samples):
    def apply(turn):
        turn.audio = to_audio(samples, start_time=turn.audio.start_time)

    return apply


def _set_content(content):
    def apply(turn):
        turn.content = content

    return apply


PROSODY_ISSUES = [
    (
        PitchVariationMetric,
        "pitch varying outside 1 to 6 semitones",
        "monotone",
        lambda: _set_audio(agent_speech(glide_hz=0)),
    ),
    (
        AbruptPitchJumpMetric,
        "an abrupt pitch jump",
        "pitch_jump",
        lambda: _set_audio(cracked_vowel(3.0)),
    ),
    (
        PitchConsistencyMetric,
        "a pitch over 4 semitones off the agent's typical pitch",
        "pitch_shift",
        lambda: _set_audio(speech_like(3.0, base_hz=230.0 * 2**0.5, seed=1)),
    ),
    (
        LoudnessConsistencyMetric,
        "a loudness over 5 dB off the agent's typical loudness",
        "loudness_shift",
        lambda: _set_audio(with_gain(agent_speech(), 2.5)),
    ),
    (
        SpeakingRateMetric,
        "a speaking rate outside 110 to 200 words per minute",
        "too_slow",
        lambda: _set_content("Hello."),
    ),
    (
        PausesMetric,
        "a pause over 2000 ms or over 50% silence",
        "long_pause",
        lambda: _set_audio(with_pause(agent_speech(), 1.5, 3.0)),
    ),
]
METRICS = [metric_class for metric_class, _, _, _ in PROSODY_ISSUES]


def _measure(metric, turns):
    metric.measure(ConversationalTestCase(turns=turns), _show_indicator=False)
    return metric


def _with_reply(index, apply, turns=None):
    turns = turns or clean_turns()
    apply(turns[index])
    return turns


@pytest.mark.parametrize("metric_class,label,_,__", PROSODY_ISSUES)
def test_a_clean_call_scores_one(metric_class, label, _, __):
    metric = _measure(metric_class(), clean_turns())
    assert metric.score == 1.0
    assert metric.reason == f"No agent reply had {label}."
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["events"] == []


@pytest.mark.parametrize("metric_class,label,event_type,build", PROSODY_ISSUES)
def test_score_is_the_share_of_agent_replies_without_the_issue(
    metric_class, label, event_type, build
):
    metric = _measure(metric_class(), _with_reply(1, build()))
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == f"1 of 3 agent replies had {label}."
    events = metric.score_breakdown["events"]
    assert {event["type"] for event in events} == {event_type}
    assert {event["turn"] for event in events} == {1}
    for event in events:
        assert event["duration_seconds"] > 0


@pytest.mark.parametrize("metric_class,_,__,build", PROSODY_ISSUES)
def test_the_callers_audio_is_ignored(metric_class, _, __, build):
    metric = _measure(metric_class(), _with_reply(0, build()))
    assert metric.score == 1.0


@pytest.mark.parametrize("metric_class", METRICS)
def test_a_call_without_agent_audio_is_skipped(metric_class):
    turns = [
        Turn(role="user", content="Hello?", audio=to_audio(speech_like(1.0))),
        Turn(role="assistant", content="Hi, how can I help?"),
    ]
    metric = _measure(metric_class(), turns)
    assert metric.score is None
    assert metric.skipped is True


@pytest.mark.parametrize("metric_class", METRICS)
def test_undecodable_agent_audio_is_skipped_not_failed(metric_class):
    turns = clean_turns(8)
    turns[1].audio = Audio.from_bytes(b"not-a-wav", "audio/wav", start_time=3.5)
    metric = _measure(metric_class(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]


def test_erratic_pitch_fails_pitch_variation():
    turns = _with_reply(1, _set_audio(agent_speech(glide_hz=90)))
    [event] = _measure(PitchVariationMetric(), turns).score_breakdown["events"]
    assert event["type"] == "erratic"
    assert event["pitch_spread_semitones"] > 6


def test_pitch_variation_limits_are_configurable():
    turns = _with_reply(1, _set_audio(agent_speech(glide_hz=90)))
    assert (
        _measure(PitchVariationMetric(max_spread_semitones=8), turns).score
        == 1.0
    )
    with pytest.raises(ValueError):
        PitchVariationMetric(min_spread_semitones=6, max_spread_semitones=6)
    with pytest.raises(ValueError):
        PitchVariationMetric(min_spread_semitones=-1)


@pytest.mark.parametrize("semitones", [8, 12, -8])
def test_a_voice_crack_is_a_pitch_jump(semitones):
    turns = _with_reply(1, _set_audio(cracked_vowel(3.0, semitones=semitones)))
    metric = _measure(AbruptPitchJumpMetric(), turns)
    assert metric.score == pytest.approx(2 / 3)


def test_a_steady_vowel_has_no_pitch_jump():
    turns = _with_reply(1, _set_audio(cracked_vowel(3.0, semitones=0)))
    assert _measure(AbruptPitchJumpMetric(), turns).score == 1.0


def test_a_crack_fails_pitch_jump_but_not_pitch_variation():
    turns = _with_reply(1, _set_audio(cracked_vowel(3.0, semitones=8)))
    assert _measure(AbruptPitchJumpMetric(), turns).score < 1.0
    assert (
        _measure(PitchVariationMetric(min_spread_semitones=0), turns).score
        == 1.0
    )


def test_pitch_consistency_limit_is_configurable():
    turns = _with_reply(
        1, _set_audio(speech_like(3.0, base_hz=230.0 * 2**0.5, seed=1))
    )
    assert (
        _measure(PitchConsistencyMetric(max_shift_semitones=7), turns).score
        == 1.0
    )
    with pytest.raises(ValueError):
        PitchConsistencyMetric(max_shift_semitones=0)


def test_loudness_consistency_limit_is_configurable():
    turns = _with_reply(1, _set_audio(with_gain(agent_speech(), 2.5)))
    assert (
        _measure(LoudnessConsistencyMetric(max_shift_db=10), turns).score == 1.0
    )
    with pytest.raises(ValueError):
        LoudnessConsistencyMetric(max_shift_db=0)


@pytest.mark.parametrize(
    "metric_class", [PitchConsistencyMetric, LoudnessConsistencyMetric]
)
def test_two_replies_are_too_few_to_compare(metric_class):
    turns = clean_turns(4)
    metric = _measure(metric_class(), turns)
    assert metric.score is None
    assert metric.reason.startswith("At least 3 ")
    assert metric.score_breakdown["eligible_turns"] == 2


def test_only_the_odd_reply_is_flagged_not_its_neighbours():
    turns = clean_turns(10)
    turns[5].audio = to_audio(
        with_gain(agent_speech(), 2.5), start_time=turns[5].audio.start_time
    )
    metric = _measure(LoudnessConsistencyMetric(), turns)
    assert [event["turn"] for event in metric.score_breakdown["events"]] == [5]


def test_a_quiet_outlier_fails_loudness_and_loudness_consistency():
    turns = _with_reply(1, _set_audio(with_gain(agent_speech(), 0.15)))
    assert _measure(LoudnessMetric(), turns).score == pytest.approx(2 / 3)
    assert _measure(LoudnessConsistencyMetric(), turns).score == pytest.approx(
        2 / 3
    )


def test_an_inaudible_reply_is_not_a_loudness_shift():
    turns = _with_reply(1, _set_audio(zeros(3.0)), clean_turns(8))
    metric = _measure(LoudnessConsistencyMetric(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]


def test_digits_are_counted_as_spoken_words():
    reply = analyze_audio(to_audio(agent_speech()))
    with_digits = speech_span_rate_wpm("Call 555-123-4567 now.", reply)
    assert with_digits == pytest.approx(12 / (reply.speech_span_seconds / 60))


def test_a_rushed_reply_is_too_fast():
    turns = _with_reply(1, _set_content("Call 555-123-4567 right now please."))
    [event] = _measure(SpeakingRateMetric(), turns).score_breakdown["events"]
    assert event["type"] == "too_fast"


def test_speaking_rate_limits_are_configurable():
    turns = _with_reply(1, _set_content("Hello."))
    assert _measure(SpeakingRateMetric(min_wpm=10), turns).score == 1.0
    with pytest.raises(ValueError):
        SpeakingRateMetric(min_wpm=200, max_wpm=110)


def test_a_reply_without_text_is_skipped_for_speaking_rate():
    turns = _with_reply(1, _set_content(""))
    metric = _measure(SpeakingRateMetric(), turns)
    assert metric.score == 1.0
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]


def test_silence_before_the_reply_is_not_a_pause():
    turns = _with_reply(1, _set_audio(with_pause(agent_speech(), 0.0, 5.0)))
    assert _measure(PausesMetric(), turns).score == 1.0


def test_a_pause_over_background_noise_is_still_a_pause():
    noisy = with_noise(with_pause(agent_speech(), 1.5, 3.0), -40)
    turns = _with_reply(1, _set_audio(noisy))
    [event] = _measure(PausesMetric(), turns).score_breakdown["events"]
    assert event["type"] == "long_pause"


def test_too_much_silence_fails_pauses():
    gappy = agent_speech()
    for at in (0.6, 1.4, 2.2):
        gappy = with_pause(gappy, at, 0.6)
    turns = _with_reply(1, _set_audio(gappy))
    [event] = _measure(PausesMetric(), turns).score_breakdown["events"]
    assert event["type"] == "too_much_silence"


def test_pause_limits_are_configurable():
    turns = _with_reply(1, _set_audio(with_pause(agent_speech(), 1.5, 3.0)))
    assert (
        _measure(
            PausesMetric(max_pause_ms=4000, max_pause_percent=90), turns
        ).score
        == 1.0
    )
    with pytest.raises(ValueError):
        PausesMetric(max_pause_ms=0)
    with pytest.raises(ValueError):
        PausesMetric(max_pause_percent=101)


def test_a_reply_too_short_for_pauses_is_skipped():
    turns = _with_reply(1, _set_audio(agent_speech(1.5)))
    metric = _measure(PausesMetric(), turns)
    assert metric.score == 1.0
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]


def test_pitch_events_hold_plain_numbers():
    turns = _with_reply(
        1, _set_audio(speech_like(3.0, base_hz=230.0 * 2**0.5, seed=1))
    )
    [event] = _measure(PitchConsistencyMetric(), turns).score_breakdown[
        "events"
    ]
    assert type(event["pitch_median_hz"]) is float
