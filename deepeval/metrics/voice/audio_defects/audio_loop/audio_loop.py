from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.metrics.voice.utils.call_activity import agent_replies
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent, placed


class AudioLoopMetric(BaseDefectMetric):
    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        replies = agent_replies(test_case)
        measurements, skipped = agent_reply_measurements(test_case)
        defects = [
            VoiceEvent(
                type="audio_loop",
                **replies.key(turn),
                description=f"Audio repeated for {duration:.1f}s",
                **placed(replies.audio[turn], offset, duration),
            )
            for turn, reply in measurements.items()
            for offset, duration in reply.loop_spans
        ]
        return DefectShare(
            list(measurements),
            defects,
            skipped,
            "looping audio",
            event_key=replies.event_key,
            timing_source=replies.timing_source,
        )

    @property
    def __name__(self):
        return "Audio Loop"
