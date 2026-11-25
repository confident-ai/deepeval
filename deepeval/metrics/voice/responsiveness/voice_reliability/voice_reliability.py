from typing import List, Optional

from deepeval.metrics.voice.base_metric import (
    BaseDefectMetric,
    BaseExchangeRollUpMetric,
)
from deepeval.metrics.voice.utils.events import VoiceEvent
from deepeval.metrics.voice.utils.roll_up import agent_audio_critical_events
from deepeval.test_case import ConversationalTestCase


class VoiceReliabilityMetric(BaseExchangeRollUpMetric):
    """Agent Responsiveness and Audio Integrity together."""

    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        use_speech_detection: bool = True,
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

    def _parts(self) -> List[BaseDefectMetric]:
        # Audio Integrity imports this package, so a top-level import is
        # circular.
        from deepeval.metrics.voice.audio_defects.audio_integrity.audio_integrity import (
            AudioIntegrityMetric,
        )
        from deepeval.metrics.voice.responsiveness.agent_responsiveness.agent_responsiveness import (
            AgentResponsivenessMetric,
        )

        parts: List[BaseDefectMetric] = []
        for part in [
            *AgentResponsivenessMetric(
                use_speech_detection=self.use_speech_detection
            )._parts(),
            *AudioIntegrityMetric(
                use_speech_detection=self.use_speech_detection
            )._parts(),
        ]:
            if all(part.__name__ != kept.__name__ for kept in parts):
                parts.append(part)
        return parts

    def _critical_events(
        self, test_case: ConversationalTestCase
    ) -> List[VoiceEvent]:
        return agent_audio_critical_events(test_case)

    @property
    def __name__(self):
        return "Voice Reliability"
