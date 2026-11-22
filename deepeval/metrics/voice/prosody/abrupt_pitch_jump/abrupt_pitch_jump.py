from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    defect_share_result,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class AbruptPitchJumpMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        measurements, skipped = agent_reply_measurements(test_case)
        eligible_turns = []
        defects = []
        for turn, reply in measurements.items():
            if not reply.voiced_pitch_frames:
                skipped.append(
                    {"turn": turn, "error": "Reply has no voiced speech."}
                )
                continue
            eligible_turns.append(turn)
            if reply.pitch_jumps:
                defects.append(
                    {
                        "type": "pitch_jump",
                        "turn": turn,
                        "pitch_jumps": reply.pitch_jumps,
                    }
                )
        return defect_share_result(
            eligible_turns,
            defects,
            skipped,
            "an abrupt pitch jump",
            no_eligible_reason="No assistant reply had voiced speech to "
            "follow its pitch.",
        )

    @property
    def __name__(self):
        return "Abrupt Pitch Jump"
