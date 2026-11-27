import pytest

from deepeval.metrics import (
    AudioClippingMetric,
    BackgroundNoiseMetric,
    LoudnessMetric,
    SilentReplyMetric,
    VolumeVarianceMetric,
)
from deepeval.test_case import Turn
from tests.test_metrics.helpers import (
    agent_speech,
    clean_turns,
    clipped,
    edit_reply,
    faded,
    measure,
    set_audio,
    speech_like,
    to_audio,
    with_gain,
    with_noise,
    with_reply_audio,
    undecodable,
    zeros,
)


def call():
    """Replies long enough to follow their volume."""
    return clean_turns(seconds=6.0)


LEVEL_ISSUES = [
    (
        BackgroundNoiseMetric,
        "background noise under 18 dB signal-to-noise",
        "background_noise",
        lambda: with_noise(agent_speech(), -30),
    ),
    (
        LoudnessMetric,
        "speech outside -32 to -10 dBFS",
        "too_quiet",
        lambda: with_gain(agent_speech(), 0.15),
    ),
    (
        VolumeVarianceMetric,
        "a volume drift over 15 dB",
        "volume_drift",
        lambda: faded(agent_speech(8.0), -20),
    ),
]


METRICS = [metric_class for metric_class, _, _, _ in LEVEL_ISSUES]


@pytest.mark.parametrize("metric_class,label,_,__", LEVEL_ISSUES)
def test_a_clean_call_scores_one(metric_class, label, _, __):
    metric = measure(metric_class(), call())
    assert metric.score == 1.0
    assert metric.reason == f"No agent reply had {label}."
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["events"] == []


@pytest.mark.parametrize("metric_class,label,event_type,build", LEVEL_ISSUES)
def test_score_is_the_share_of_agent_replies_without_the_issue(
    metric_class, label, event_type, build
):
    metric = measure(metric_class(), edit_reply(1, set_audio(build()), call()))
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == f"1 of 3 agent replies had {label}."
    [event] = metric.score_breakdown["events"]
    assert event["type"] == event_type
    assert event["turn"] == 1
    assert event["mark"] == "whole_reply"
    assert event["offset_seconds"] == 0.0
    assert event["timing_source"] == "turn_timestamps"
    assert event["description"]


@pytest.mark.parametrize("metric_class,_,__,build", LEVEL_ISSUES)
def test_the_callers_audio_is_ignored(metric_class, _, __, build):
    metric = measure(metric_class(), edit_reply(0, set_audio(build()), call()))
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
    turns = call()
    undecodable(turns[1])
    metric = measure(metric_class(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]


@pytest.mark.parametrize(
    "metric_class,error",
    [
        (BackgroundNoiseMetric, "Reply has no audible sound to measure."),
        (LoudnessMetric, "Reply has under 1 second of speech."),
    ],
)
def test_an_inaudible_reply_is_left_to_silent_reply(metric_class, error):
    turns = with_reply_audio(1, zeros(3.0))
    metric = measure(metric_class(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["skipped"] == [{"turn": 1, "error": error}]
    assert measure(SilentReplyMetric(), turns).score == pytest.approx(2 / 3)


def test_a_loud_clipped_reply_fails_loudness_and_clipping():
    turns = with_reply_audio(1, clipped(3.0))
    loudness = measure(LoudnessMetric(), turns)
    assert [event["type"] for event in loudness.score_breakdown["events"]] == [
        "too_loud"
    ]
    assert measure(AudioClippingMetric(), turns).score == pytest.approx(2 / 3)
