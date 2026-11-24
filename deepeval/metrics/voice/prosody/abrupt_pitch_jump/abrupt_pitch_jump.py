from deepeval.metrics.voice.utils.defects import (
    agent_reply_measurements,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.analysis import PITCH_FRAME_MS
from deepeval.metrics.voice.utils.events import MOMENT, VoiceEvent, placed

PITCH_FRAME_SECONDS = PITCH_FRAME_MS / 1000


class AbruptPitchJumpMetric(BaseDefectMetric):
    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
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
            for at in reply.pitch_jump_times:
                # The step sits between two 40 ms frames; mark both.
                defects.append(
                    VoiceEvent(
                        type="pitch_jump",
                        turn=turn,
                        mark=MOMENT,
                        description="Pitch jumped abruptly",
                        **placed(
                            test_case.turns[turn].audio,
                            max(0.0, at - PITCH_FRAME_SECONDS),
                            2 * PITCH_FRAME_SECONDS,
                        ),
                    )
                )
        return DefectShare(
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
