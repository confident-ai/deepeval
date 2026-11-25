from typing import List, Optional, Type, Union

from deepeval.metrics.voice.audio_defects.abrupt_cutoff.abrupt_cutoff import (
    AbruptCutoffMetric,
)
from deepeval.metrics.voice.audio_defects.audio_clipping.audio_clipping import (
    AudioClippingMetric,
)
from deepeval.metrics.voice.audio_defects.dead_stream.dead_stream import (
    DeadStreamMetric,
)
from deepeval.metrics.voice.audio_defects.held_tone.held_tone import (
    HeldToneMetric,
)
from deepeval.metrics.voice.base_metric import (
    BaseDefectMetric,
    BaseRollUpMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.templates import make_template_class


AudioGlitchTemplate = make_template_class("AudioGlitchMetric")


class AudioGlitchMetric(BaseRollUpMetric):
    # model, async_mode and evaluation_template are accepted for code written
    # when an LLM wrote this metric's reason; the reason is now fixed text.
    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        model: Optional[Union[str, DeepEvalBaseLLM]] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        flaky: bool = False,
        evaluation_template: Type[AudioGlitchTemplate] = AudioGlitchTemplate,
        use_speech_detection: bool = True,
    ):
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.use_speech_detection = use_speech_detection

    def _parts(self) -> List[BaseDefectMetric]:
        return [
            HeldToneMetric(),
            DeadStreamMetric(),
            AbruptCutoffMetric(use_speech_detection=self.use_speech_detection),
            AudioClippingMetric(),
        ]

    @property
    def __name__(self):
        return "Audio Glitch"
