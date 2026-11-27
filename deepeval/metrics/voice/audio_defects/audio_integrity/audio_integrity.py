from typing import List, Optional, Union

from deepeval.metrics.voice.audio_defects.abrupt_cutoff.abrupt_cutoff import (
    AbruptCutoffMetric,
)
from deepeval.metrics.voice.audio_defects.audio_clipping.audio_clipping import (
    AudioClippingMetric,
)
from deepeval.metrics.voice.audio_defects.audio_dropout.audio_dropout import (
    AudioDropoutMetric,
)
from deepeval.metrics.voice.audio_defects.audio_loop.audio_loop import (
    AudioLoopMetric,
)
from deepeval.metrics.voice.base_metric import (
    BaseDefectMetric,
    BaseRollUpMetric,
)
from deepeval.metrics.voice.responsiveness.silent_reply.silent_reply import (
    SilentReplyMetric,
)
from deepeval.metrics.voice.utils.events import VoiceEvent
from deepeval.metrics.voice.utils.roll_up import agent_audio_critical_events
from deepeval.models.base_model import DeepEvalBaseSTT
from deepeval.models.speech_selection import initialize_stt_model
from deepeval.test_case import ConversationalTestCase


class AudioIntegrityMetric(BaseRollUpMetric):
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        use_speech_detection: bool = True,
        stt_model: Optional[Union[str, DeepEvalBaseSTT]] = None,
        flaky: bool = False,
    ):
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.use_speech_detection = use_speech_detection
        self.stt_model = (
            stt_model
            if stt_model is None or isinstance(stt_model, DeepEvalBaseSTT)
            else initialize_stt_model(stt_model)
        )

    def _parts(self) -> List[BaseDefectMetric]:
        return [
            AbruptCutoffMetric(use_speech_detection=self.use_speech_detection),
            AudioLoopMetric(),
            AudioDropoutMetric(),
            AudioClippingMetric(),
            SilentReplyMetric(
                use_speech_detection=self.use_speech_detection,
                stt_model=self.stt_model,
            ),
        ]

    def _critical_events(
        self, test_case: ConversationalTestCase
    ) -> List[VoiceEvent]:
        return agent_audio_critical_events(test_case)

    @property
    def __name__(self):
        return "Audio Integrity"
