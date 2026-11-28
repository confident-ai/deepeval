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
from deepeval.test_case import Audio, Turn
from tests.test_metrics.helpers import (
    agent_speech,
    clean_turns,
    cracked_vowel,
    edit_reply,
    measure,
    set_audio,
    speech_like,
    to_audio,
    with_gain,
    with_pause,
)


def call(count=6):
    """Replies that flow fast enough to read a speaking rate from, and long
    enough to read a base pitch from."""
    return clean_turns(count, seconds=6.0, rhythm=2.5)


PROSODY_ISSUES = [
    (
        PitchVariationMetric,
        "pitch varying under 1 semitones",
        "monotone",
        lambda: set_audio(agent_speech(5.0, glide_hz=0)),
    ),
    (
        AbruptPitchJumpMetric,
        "an abrupt pitch jump",
        "pitch_jump",
        lambda: set_audio(cracked_vowel(3.0)),
    ),
    (
        PitchConsistencyMetric,
        "a pitch over 3 semitones off the agent's typical pitch",
        "pitch_shift",
        lambda: set_audio(
            speech_like(6.0, base_hz=230.0 * 2**0.5, seed=1, rhythm=2.5)
        ),
    ),
    (
        LoudnessConsistencyMetric,
        "a loudness over 7 dB off the agent's typical loudness",
        "loudness_shift",
        lambda: set_audio(with_gain(agent_speech(), 2.5)),
    ),
    (
        SpeakingRateMetric,
        "a speaking rate outside 3.5 to 10 syllables a second",
        "too_slow",
        lambda: set_audio(agent_speech(rhythm=1.5)),
    ),
    (
        PausesMetric,
        "a pause over 4000 ms or over 50% silence",
        "long_pause",
        lambda: set_audio(with_pause(agent_speech(), 1.5, 5.0)),
    ),
]


METRICS = [metric_class for metric_class, _, _, _ in PROSODY_ISSUES]


@pytest.mark.parametrize("metric_class,label,_,__", PROSODY_ISSUES)
def test_a_clean_call_scores_one(metric_class, label, _, __):
    metric = measure(metric_class(), call())
    assert metric.score == 1.0
    assert metric.reason == f"No agent reply had {label}."
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["events"] == []


@pytest.mark.parametrize("metric_class,label,event_type,build", PROSODY_ISSUES)
def test_score_is_the_share_of_agent_replies_without_the_issue(
    metric_class, label, event_type, build
):
    metric = measure(metric_class(), edit_reply(1, build(), call()))
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == f"1 of 3 agent replies had {label}."
    events = metric.score_breakdown["events"]
    assert {event["type"] for event in events} == {event_type}
    assert {event["turn"] for event in events} == {1}
    for event in events:
        assert event["duration_seconds"] > 0


@pytest.mark.parametrize("metric_class,_,__,build", PROSODY_ISSUES)
def test_the_callers_audio_is_ignored(metric_class, _, __, build):
    metric = measure(metric_class(), edit_reply(0, build(), call()))
    assert metric.score == 1.0


@pytest.mark.parametrize("metric_class", METRICS)
def test_a_call_without_agent_audio_is_skipped(metric_class):
    turns = [
        Turn(role="user", content="Hello?", audio=to_audio(speech_like(1.0))),
        Turn(role="assistant", content="Hi, how can I help?"),
    ]
    metric = measure(metric_class(), turns)
    assert metric.score is None
    assert metric.skipped is True


@pytest.mark.parametrize("metric_class", METRICS)
def test_undecodable_agent_audio_is_skipped_not_failed(metric_class):
    turns = call(8)
    turns[1].audio = Audio.from_bytes(b"not-a-wav", "audio/wav", start_time=3.5)
    metric = measure(metric_class(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 3
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]


def test_a_crack_fails_pitch_jump_but_not_pitch_variation():
    turns = edit_reply(1, set_audio(cracked_vowel(3.0, semitones=8)))
    assert measure(AbruptPitchJumpMetric(), turns).score < 1.0
    assert (
        measure(PitchVariationMetric(min_spread_semitones=0), turns).score
        == 1.0
    )


@pytest.mark.parametrize(
    "metric_class", [PitchConsistencyMetric, LoudnessConsistencyMetric]
)
def test_two_replies_are_too_few_to_compare(metric_class):
    metric = measure(metric_class(), call(4))
    assert metric.score is None
    assert metric.reason.startswith("At least 3 ")
    assert metric.score_breakdown["eligible_turns"] == 2


def test_a_quiet_outlier_fails_loudness_and_loudness_consistency():
    turns = edit_reply(1, set_audio(with_gain(agent_speech(), 0.15)))
    assert measure(LoudnessMetric(), turns).score == pytest.approx(2 / 3)
    assert measure(LoudnessConsistencyMetric(), turns).score == pytest.approx(
        2 / 3
    )
