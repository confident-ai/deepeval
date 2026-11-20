from deepeval.metrics.voice._defects import (
    agent_reply_measurements,
    defect_share_result,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class AudioLoopMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        measurements, skipped = agent_reply_measurements(test_case)
        defects = [
            {"type": "audio_loop", "turn": turn, "count": reply.loop_events}
            for turn, reply in measurements.items()
            if reply.loop_events
        ]
        return defect_share_result(
            list(measurements), defects, skipped, "looping audio"
        )

    @property
    def __name__(self):
        return "Audio Loop"
