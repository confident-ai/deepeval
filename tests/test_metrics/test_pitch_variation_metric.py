import pytest

from deepeval.metrics import PitchVariationMetric
from tests.test_metrics.helpers import (
    agent_speech,
    edit_reply,
    measure,
    set_audio,
)


def test_erratic_pitch_fails_pitch_variation():
    turns = edit_reply(1, set_audio(agent_speech(glide_hz=90)))
    [event] = measure(PitchVariationMetric(), turns).score_breakdown["events"]
    assert event["type"] == "erratic"
    assert event["pitch_spread_semitones"] > 6


def test_pitch_variation_limits_are_configurable():
    turns = edit_reply(1, set_audio(agent_speech(glide_hz=90)))
    assert (
        measure(PitchVariationMetric(max_spread_semitones=8), turns).score
        == 1.0
    )
    with pytest.raises(ValueError):
        PitchVariationMetric(min_spread_semitones=6, max_spread_semitones=6)
    with pytest.raises(ValueError):
        PitchVariationMetric(min_spread_semitones=-1)
