import pytest

from deepeval.metrics import (
    AudioClippingMetric,
    BackgroundNoiseMetric,
    LoudnessMetric,
    SilentReplyMetric,
    SpeechIntelligibilityMetric,
    VolumeVarianceMetric,
)
from deepeval.metrics.voice.utils.analysis import analyze_audio
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from tests.test_metrics.helpers import (
    clean_turns,
    clipped,
    faded,
    speech_like,
    to_audio,
    with_gain,
    with_noise,
    zeros,
)


def agent_speech(seconds=3.0):
    return speech_like(seconds, base_hz=230.0, seed=1)


LEVEL_ISSUES = [
    (
        BackgroundNoiseMetric,
        "background noise under 20 dB signal-to-noise",
        "background_noise",
        lambda: with_noise(agent_speech(), -30),
    ),
    (
        LoudnessMetric,
        "speech outside -32 to -6 dBFS",
        "too_quiet",
        lambda: with_gain(agent_speech(), 0.15),
    ),
    (
        VolumeVarianceMetric,
        "a volume drift over 10 dB",
        "volume_drift",
        lambda: faded(agent_speech(), -20),
    ),
]
METRICS = [metric_class for metric_class, _, _, _ in LEVEL_ISSUES]


def _measure(metric, turns):
    metric.measure(ConversationalTestCase(turns=turns), _show_indicator=False)
    return metric


def _with_reply(index, samples, turns=None):
    turns = turns or clean_turns()
    turns[index].audio = to_audio(samples, start_time=index * 3.5)
    return turns


@pytest.mark.parametrize("metric_class,label,_,__", LEVEL_ISSUES)
def test_a_clean_call_scores_one(metric_class, label, _, __):
    metric = _measure(metric_class(), clean_turns())
    assert metric.score == 1.0
    assert metric.reason == f"No agent reply had {label}."
    assert metric.score_breakdown["eligible_turns"] == 3
    assert metric.score_breakdown["events"] == []


@pytest.mark.parametrize("metric_class,label,event_type,build", LEVEL_ISSUES)
def test_score_is_the_share_of_agent_replies_without_the_issue(
    metric_class, label, event_type, build
):
    metric = _measure(metric_class(), _with_reply(1, build()))
    assert metric.score == pytest.approx(2 / 3)
    assert metric.reason == f"1 of 3 agent replies had {label}."
    [event] = metric.score_breakdown["events"]
    assert event["type"] == event_type
    assert event["turn"] == 1


@pytest.mark.parametrize("metric_class,_,__,build", LEVEL_ISSUES)
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
    turns = clean_turns()
    turns[1].audio = Audio.from_bytes(b"not-a-wav", "audio/wav", start_time=3.5)
    metric = _measure(metric_class(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]


@pytest.mark.parametrize(
    "metric_class", [BackgroundNoiseMetric, LoudnessMetric]
)
def test_an_inaudible_reply_is_left_to_silent_reply(metric_class):
    turns = _with_reply(1, zeros(3.0))
    metric = _measure(metric_class(), turns)
    assert metric.score == 1.0
    assert metric.score_breakdown["skipped"] == [
        {"turn": 1, "error": "Reply has no audible sound to measure."}
    ]
    assert _measure(SilentReplyMetric(), turns).score == pytest.approx(2 / 3)


def test_noise_louder_than_the_silence_cutoff_still_lowers_the_snr():
    snrs = [
        analyze_audio(
            to_audio(with_noise(agent_speech(), level))
        ).estimated_snr_db
        for level in (-50, -40, -30, -25)
    ]
    assert snrs == sorted(snrs, reverse=True)
    assert snrs[-1] < 10


def test_loud_noise_lowers_speech_intelligibility():
    quiet = _measure(
        SpeechIntelligibilityMetric(),
        _with_reply(1, with_noise(agent_speech(), -50)),
    )
    loud = _measure(
        SpeechIntelligibilityMetric(),
        _with_reply(1, with_noise(agent_speech(), -30)),
    )
    assert loud.score < quiet.score


def test_background_noise_limit_is_configurable():
    turns = _with_reply(1, with_noise(agent_speech(), -30))
    assert _measure(BackgroundNoiseMetric(min_snr_db=10), turns).score == 1.0


def test_a_negative_snr_limit_is_rejected():
    with pytest.raises(ValueError):
        BackgroundNoiseMetric(min_snr_db=-1)


def test_a_loud_clipped_reply_fails_loudness_and_clipping():
    turns = _with_reply(1, clipped(3.0))
    loudness = _measure(LoudnessMetric(), turns)
    assert [event["type"] for event in loudness.score_breakdown["events"]] == [
        "too_loud"
    ]
    assert _measure(AudioClippingMetric(), turns).score == pytest.approx(2 / 3)


def test_loudness_limits_are_configurable():
    loud = _with_reply(1, with_gain(agent_speech(), 3))
    assert _measure(LoudnessMetric(), loud).score == 1.0
    strict = _measure(LoudnessMetric(min_dbfs=-40, max_dbfs=-12), loud)
    assert [event["type"] for event in strict.score_breakdown["events"]] == [
        "too_loud"
    ]
    assert (
        strict.reason
        == "1 of 3 agent replies had speech outside -40 to -12 dBFS."
    )
    quiet = _with_reply(1, with_gain(agent_speech(), 0.05))
    assert _measure(LoudnessMetric(min_dbfs=-60), quiet).score == 1.0


@pytest.mark.parametrize("min_dbfs,max_dbfs", [(-6, -6), (-6, -32), (-32, 3)])
def test_invalid_loudness_limits_are_rejected(min_dbfs, max_dbfs):
    with pytest.raises(ValueError):
        LoudnessMetric(min_dbfs=min_dbfs, max_dbfs=max_dbfs)


@pytest.mark.parametrize("db", [-20, 20])
def test_a_fade_or_swell_is_a_volume_drift(db):
    metric = _measure(
        VolumeVarianceMetric(), _with_reply(1, faded(agent_speech(6.0), db))
    )
    [event] = metric.score_breakdown["events"]
    assert event["drift_db"] * db > 0
    assert abs(event["drift_db"]) > 10


def test_steady_speech_has_no_volume_drift():
    metric = _measure(VolumeVarianceMetric(), _with_reply(1, agent_speech(6.0)))
    assert metric.score == 1.0


def test_volume_drift_limit_is_configurable():
    turns = _with_reply(1, faded(agent_speech(6.0), -20))
    assert _measure(VolumeVarianceMetric(max_drift_db=20), turns).score == 1.0
    with pytest.raises(ValueError):
        VolumeVarianceMetric(max_drift_db=0)


def test_a_reply_too_short_to_follow_is_skipped():
    metric = _measure(VolumeVarianceMetric(), _with_reply(1, agent_speech(1.5)))
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]


def test_a_call_of_only_short_replies_is_skipped():
    turns = clean_turns()
    for index in (1, 3, 5):
        turns[index].audio = to_audio(agent_speech(1.5))
    metric = _measure(VolumeVarianceMetric(), turns)
    assert metric.score is None
    assert metric.reason == (
        "No assistant reply was long enough to follow its volume."
    )


def test_a_reply_too_short_to_measure_noise_is_skipped():
    metric = _measure(
        BackgroundNoiseMetric(),
        _with_reply(1, with_noise(agent_speech(1.5), -30)),
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]


def test_short_clips_keep_the_earlier_snr_estimate():
    clip = with_noise(agent_speech(1.5), -30)
    # Without pauses quieter than the silence cutoff, the earlier estimate
    # assumes 40 dB; the composite metrics keep scoring short clips that way.
    assert analyze_audio(to_audio(clip)).estimated_snr_db == pytest.approx(40.0)
