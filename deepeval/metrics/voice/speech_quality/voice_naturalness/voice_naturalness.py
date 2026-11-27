from typing import List

from deepeval.metrics.voice.audio_defects.audio_clipping.audio_clipping import (
    AudioClippingMetric,
)
from deepeval.metrics.voice.audio_defects.audio_dropout.audio_dropout import (
    AudioDropoutMetric,
)
from deepeval.metrics.voice.audio_defects.audio_loop.audio_loop import (
    AudioLoopMetric,
)
from deepeval.metrics.voice.audio_level.background_noise.background_noise import (
    BackgroundNoiseMetric,
)
from deepeval.metrics.voice.base_metric import (
    BaseDefectMetric,
    BaseRollUpMetric,
)
from deepeval.metrics.voice.prosody.pauses.pauses import PausesMetric
from deepeval.metrics.voice.prosody.pitch_variation.pitch_variation import (
    PitchVariationMetric,
)
from deepeval.metrics.voice.prosody.speaking_rate.speaking_rate import (
    SpeakingRateMetric,
)


class VoiceNaturalnessMetric(BaseRollUpMetric):
    def _parts(self) -> List[BaseDefectMetric]:
        return [
            AudioClippingMetric(),
            AudioDropoutMetric(),
            AudioLoopMetric(),
            PausesMetric(),
            BackgroundNoiseMetric(),
            SpeakingRateMetric(),
            PitchVariationMetric(),
        ]

    @property
    def __name__(self):
        return "Voice Naturalness"
