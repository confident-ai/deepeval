import pytest

from deepeval.metrics import PitchConsistencyMetric
from tests.test_metrics.helpers import (
    edit_reply,
    measure,
    set_audio,
    speech_like,
)


def test_pitch_consistency_limit_is_configurable():
    turns = edit_reply(
        1, set_audio(speech_like(3.0, base_hz=230.0 * 2**0.5, seed=1))
    )
    assert (
        measure(PitchConsistencyMetric(max_shift_semitones=7), turns).score
        == 1.0
    )
    with pytest.raises(ValueError):
        PitchConsistencyMetric(max_shift_semitones=0)


def test_pitch_events_hold_plain_numbers():
    turns = edit_reply(
        1, set_audio(speech_like(3.0, base_hz=230.0 * 2**0.5, seed=1))
    )
    [event] = measure(PitchConsistencyMetric(), turns).score_breakdown["events"]
    assert type(event["pitch_median_hz"]) is float
