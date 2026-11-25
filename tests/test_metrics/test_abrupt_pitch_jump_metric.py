import pytest

from deepeval.metrics import AbruptPitchJumpMetric
from tests.test_metrics.helpers import (
    cracked_vowel,
    edit_reply,
    measure,
    set_audio,
)


@pytest.mark.parametrize("semitones", [8, 12, -8])
def test_a_voice_crack_is_a_pitch_jump(semitones):
    turns = edit_reply(1, set_audio(cracked_vowel(3.0, semitones=semitones)))
    metric = measure(AbruptPitchJumpMetric(), turns)
    assert metric.score == pytest.approx(2 / 3)


def test_a_steady_vowel_has_no_pitch_jump():
    turns = edit_reply(1, set_audio(cracked_vowel(3.0, semitones=0)))
    assert measure(AbruptPitchJumpMetric(), turns).score == 1.0
