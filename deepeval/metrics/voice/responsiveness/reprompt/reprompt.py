from deepeval.metrics.voice.utils.call_activity import caller_lines
from deepeval.metrics.voice.utils.defects import (
    CALLER_LINES,
    NO_CALL_AUDIO_REASON,
    call_has_audio,
    DefectShare,
)
from deepeval.metrics.voice.utils.detectors import response_was_owed
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent


class RepromptMetric(BaseDefectMetric):
    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        defect_label = "to be repeated before the agent replied"
        if not call_has_audio(test_case):
            return DefectShare(
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
        for first, last in caller_lines(turns):
            if not response_was_owed(turns[last].content):
                continue
            eligible_turns.append(first)
            next_turn = turns[last + 1] if last + 1 < len(turns) else None
            # A line never answered at all is Failed Response's to judge.
            answered_later = any(
                later.role == "assistant" for later in turns[last + 1 :]
            )
            if (
                next_turn is not None
                and next_turn.role == "user"
                and answered_later
            ):
                defects.append(
                    VoiceEvent(
                        type="user_reprompted",
                        turn=first,
                        reprompt_turn=last + 1,
                    )
                )
        return DefectShare(
            eligible_turns,
            defects,
            [],
            defect_label,
            no_eligible_reason="No caller line needed a reply.",
            counted=CALLER_LINES,
        )

    @property
    def __name__(self):
        return "Reprompt"
