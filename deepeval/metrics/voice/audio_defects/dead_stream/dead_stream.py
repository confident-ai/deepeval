from deepeval.metrics.voice.utils.defects import (
    agent_glitch_events,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase


class DeadStreamMetric(BaseDefectMetric):
    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        eligible_turns, events, skipped = agent_glitch_events(test_case)
        defects = [
            event.model_copy(
                update={
                    "description": "Agent's audio went dead for "
                    f"{event.duration_seconds:.1f}s"
                }
            )
            for event in events
            if event.type == "dead_stream"
        ]
        return DefectShare(eligible_turns, defects, skipped, "a dead stream")

    @property
    def __name__(self):
        return "Dead Stream"
