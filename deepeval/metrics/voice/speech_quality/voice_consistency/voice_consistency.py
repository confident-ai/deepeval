from typing import List

from deepeval.metrics.voice.audio_level.loudness_consistency.loudness_consistency import (
    LoudnessConsistencyMetric,
)
from deepeval.metrics.voice.base_metric import (
    BaseDefectMetric,
    BaseRollUpMetric,
)
from deepeval.metrics.voice.prosody.pitch_consistency.pitch_consistency import (
    PitchConsistencyMetric,
)


class VoiceConsistencyMetric(BaseRollUpMetric):
    def _parts(self) -> List[BaseDefectMetric]:
        return [PitchConsistencyMetric(), LoudnessConsistencyMetric()]

    @property
    def __name__(self):
        return "Voice Consistency"
