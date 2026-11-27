from deepeval.metrics.voice.utils.defects import (
    agent_glitch_events,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.metrics.voice.utils.call_activity import agent_replies
from deepeval.test_case import ConversationalTestCase


class AudioClippingMetric(BaseDefectMetric):
    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        replies = agent_replies(test_case)
        eligible_turns, events, skipped = agent_glitch_events(test_case)
        defects = [
            event.model_copy(update={"description": "Audio was clipped"})
            for event in events
            if event.type == "clipping"
        ]
        return DefectShare(
            eligible_turns,
            defects,
            skipped,
            "clipped audio",
            event_key=replies.event_key,
            timing_source=replies.timing_source,
        )

    @property
    def __name__(self):
        return "Audio Clipping"
