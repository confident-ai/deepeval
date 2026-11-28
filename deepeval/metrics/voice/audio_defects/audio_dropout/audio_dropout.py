from typing import Optional

from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.config.voice_eval_mode import VoiceEvalModeName
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent, placed


class AudioDropoutMetric(BaseDefectMetric):
    event_question = "Does the speaker's audio suddenly drop out or go silent mid-word, then come back?"
    event_speaker = "assistant"

    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        audio_decision_model: Optional[DeepEvalBaseAudioDecisionModel] = None,
        eval_mode: Optional[VoiceEvalModeName] = None,
        flaky: bool = False,
    ):
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self._use_audio_decision_model(audio_decision_model, eval_mode)

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        measurements, skipped = agent_reply_measurements(test_case)
        defects = [
            VoiceEvent(
                type="audio_dropout",
                turn=turn,
                description=f"Audio dropped out for {duration * 1000:.0f} ms",
                **placed(test_case.turns[turn].audio, offset, duration),
            )
            for turn, reply in measurements.items()
            for offset, duration in reply.dropout_spans
        ]
        return DefectShare(
            list(measurements), defects, skipped, "audio dropouts"
        )

    @property
    def __name__(self):
        return "Audio Dropout"
