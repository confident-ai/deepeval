from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    defect_share_result,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent, placed


class AudioDropoutMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        measurements, skipped = agent_reply_measurements(test_case)
        defects = [
            VoiceEvent(
                type="audio_dropout",
                turn=turn,
                **placed(test_case.turns[turn].audio, offset, duration),
            )
            for turn, reply in measurements.items()
            for offset, duration in reply.dropout_spans
        ]
        return defect_share_result(
            list(measurements), defects, skipped, "audio dropouts"
        )

    @property
    def __name__(self):
        return "Audio Dropout"
