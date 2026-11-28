from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent, placed


class AudioDropoutMetric(BaseDefectMetric):
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
