from deepeval.metrics.voice._defects import (
    agent_glitch_events,
    agent_reply_measurements,
    defect_share_result,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class AbruptCutoffMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        measurements, skipped = agent_reply_measurements(test_case)
        _, events, _ = agent_glitch_events(test_case)
        stream_cuts = {
            event["turn"]: event
            for event in events
            if event["type"] == "stream_cut"
        }
        defects = [
            stream_cuts.get(turn, {"type": "abrupt_cutoff", "turn": turn})
            for turn, reply in measurements.items()
            if (turn in stream_cuts or reply.ends_abruptly)
            and not test_case.turns[turn].interrupted
        ]
        return defect_share_result(
            list(measurements), defects, skipped, "an abrupt cut-off"
        )

    @property
    def __name__(self):
        return "Abrupt Cut-off"
