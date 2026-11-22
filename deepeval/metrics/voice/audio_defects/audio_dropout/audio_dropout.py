from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    defect_share_result,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class AudioDropoutMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        measurements, skipped = agent_reply_measurements(test_case)
        defects = [
            {
                "type": "audio_dropout",
                "turn": turn,
                "count": reply.dropout_events,
            }
            for turn, reply in measurements.items()
            if reply.dropout_events
        ]
        return defect_share_result(
            list(measurements), defects, skipped, "audio dropouts"
        )

    @property
    def __name__(self):
        return "Audio Dropout"
