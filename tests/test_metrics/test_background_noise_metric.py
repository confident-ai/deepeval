import pytest

from deepeval.metrics import BackgroundNoiseMetric
from deepeval.metrics.voice.utils.analysis import analyze_audio
from tests.test_metrics.system_one_fakes import FakeAudioDecisionModel
from tests.test_metrics.helpers import (
    agent_speech,
    clip_shape,
    measure,
    to_audio,
    with_noise,
    with_reply_audio,
)


def test_noise_louder_than_the_silence_cutoff_still_lowers_the_snr():
    snrs = [
        analyze_audio(
            to_audio(with_noise(agent_speech(), level))
        ).estimated_snr_db
        for level in (-50, -40, -30, -25)
    ]
    assert snrs == sorted(snrs, reverse=True)
    assert snrs[-1] < 10


def test_background_noise_limit_is_configurable():
    turns = with_reply_audio(1, with_noise(agent_speech(), -30))
    assert measure(BackgroundNoiseMetric(min_snr_db=10), turns).score == 1.0


def test_a_negative_snr_limit_is_rejected():
    with pytest.raises(ValueError):
        BackgroundNoiseMetric(min_snr_db=-1)


def test_a_reply_too_short_to_measure_noise_is_skipped():
    metric = measure(
        BackgroundNoiseMetric(),
        with_reply_audio(1, with_noise(agent_speech(1.5), -30)),
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["eligible_turns"] == 2
    assert [entry["turn"] for entry in metric.score_breakdown["skipped"]] == [1]


def test_short_clips_keep_the_earlier_snr_estimate():
    clip = with_noise(agent_speech(1.5), -30)
    # Without pauses quieter than the silence cutoff, the earlier estimate
    # assumes 40 dB; the composite metrics keep scoring short clips that way.
    assert analyze_audio(to_audio(clip)).estimated_snr_db == pytest.approx(40.0)


def test_noise_the_decision_model_does_not_hear_is_dismissed():
    model = FakeAudioDecisionModel(probability=0.1)
    turns = with_reply_audio(1, with_noise(agent_speech(), -30))
    metric = measure(
        BackgroundNoiseMetric(eval_mode="hybrid", audio_decision_model=model),
        turns,
    )
    assert metric.score == 1.0
    assert metric.score_breakdown["dismissed_events"] == 1
    # The whole reply it was measured on.
    [clip] = model.clips
    assert clip_shape(clip)[1] == pytest.approx(3.0, abs=0.05)
