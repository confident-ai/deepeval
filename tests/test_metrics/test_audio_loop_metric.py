import pytest

from deepeval.metrics import AudioLoopMetric
from tests.test_metrics.helpers import (
    loop_clip,
    measure,
    with_defect,
    with_noise,
)


def test_a_loop_is_found_after_the_audio_is_re_encoded():
    turns = with_defect(1, lambda: with_noise(loop_clip(), -40))
    metric = measure(AudioLoopMetric(), turns)
    assert metric.score == pytest.approx(2 / 3)
