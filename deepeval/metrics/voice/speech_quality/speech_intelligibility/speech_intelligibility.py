from typing import List

from deepeval.metrics.voice.audio_defects.audio_clipping.audio_clipping import (
    AudioClippingMetric,
)
from deepeval.metrics.voice.audio_defects.audio_dropout.audio_dropout import (
    AudioDropoutMetric,
)
from deepeval.metrics.voice.audio_level.background_noise.background_noise import (
    BackgroundNoiseMetric,
)
from deepeval.metrics.voice.audio_level.loudness.loudness import LoudnessMetric
from deepeval.metrics.voice.base_metric import (
    BaseDefectMetric,
    BaseRollUpMetric,
)


class SpeechIntelligibilityMetric(BaseRollUpMetric):
    def _parts(self) -> List[BaseDefectMetric]:
        return [
            BackgroundNoiseMetric(),
            LoudnessMetric(),
            AudioClippingMetric(),
            AudioDropoutMetric(),
        ]

    @property
    def __name__(self):
        return "Speech Intelligibility"
