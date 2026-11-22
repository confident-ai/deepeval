from deepeval.metrics.voice.utils.defects import (
    agent_glitch_events,
    defect_share_result,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class HeldToneMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        eligible_turns, events, skipped = agent_glitch_events(test_case)
        defects = [event for event in events if event.type == "held_tone"]
        return defect_share_result(
            eligible_turns, defects, skipped, "a held tone"
        )

    @property
    def __name__(self):
        return "Held Tone"
