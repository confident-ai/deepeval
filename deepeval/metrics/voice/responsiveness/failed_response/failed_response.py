from deepeval.metrics.voice.utils.defects import (
    NO_CALL_AUDIO_REASON,
    call_has_audio,
    defect_share_result,
)
from deepeval.metrics.voice.utils.detectors import response_was_owed
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent

CALLER_LINES = ("caller line", "caller lines")


class FailedResponseMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        defect_label = "no reply from the agent"
        if not call_has_audio(test_case):
            return defect_share_result(
                [],
                [],
                [],
                defect_label,
                no_eligible_reason=NO_CALL_AUDIO_REASON,
                counted=CALLER_LINES,
            )
        turns = test_case.turns
        eligible_turns = []
        defects = []
        for index, turn in enumerate(turns):
            if turn.role != "user" or not response_was_owed(turn.content):
                continue
            eligible_turns.append(index)
            reply_turn = next(
                (
                    later
                    for later in range(index + 1, len(turns))
                    if turns[later].role == "assistant"
                ),
                None,
            )
            if reply_turn is None:
                defects.append(
                    VoiceEvent(type="call_ended_without_reply", turn=index)
                )
            elif (
                not turns[reply_turn].content.strip()
                and turns[reply_turn].audio is None
            ):
                defects.append(
                    VoiceEvent(
                        type="empty_reply", turn=index, reply_turn=reply_turn
                    )
                )
        return defect_share_result(
            eligible_turns,
            defects,
            [],
            defect_label,
            no_eligible_reason="No caller line needed a reply.",
            counted=CALLER_LINES,
        )

    @property
    def __name__(self):
        return "Failed Response"
