from deepeval.metrics.voice.utils.defects import (
    agent_glitch_events,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase


class AudioClippingMetric(BaseDefectMetric):
    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        eligible_turns, events, skipped = agent_glitch_events(test_case)
        defects = [
            event.model_copy(update={"description": "Audio was clipped"})
            for event in events
            if event.type == "clipping"
        ]
        return DefectShare(eligible_turns, defects, skipped, "clipped audio")

    @property
    def __name__(self):
        return "Audio Clipping"
