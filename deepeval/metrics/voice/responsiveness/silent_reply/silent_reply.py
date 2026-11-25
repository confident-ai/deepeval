from deepeval.metrics.voice.utils.defects import (
    NO_CALL_AUDIO_REASON,
    agent_reply_measurements,
    call_has_audio,
    DefectShare,
    is_inaudible,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent


class SilentReplyMetric(BaseDefectMetric):
    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        defect_label = "a reply the caller could not hear"
        if not call_has_audio(test_case):
            return DefectShare(
                [],
                [],
                [],
                defect_label,
                no_eligible_reason=NO_CALL_AUDIO_REASON,
            )
        measurements, skipped = agent_reply_measurements(test_case)
        undecodable = {entry["turn"] for entry in skipped}
        eligible_turns = []
        defects = []
        for index, turn in enumerate(test_case.turns):
            if (
                turn.role != "assistant"
                or index in undecodable
                or (not turn.content.strip() and turn.audio is None)
            ):
                continue
            eligible_turns.append(index)
            if turn.audio is None:
                defects.append(
                    VoiceEvent(
                        type="no_audio",
                        turn=index,
                        description="Reply had no audio",
                    )
                )
            elif is_inaudible(measurements[index]):
                defects.append(
                    VoiceEvent(
                        type="no_speech",
                        turn=index,
                        description="Reply had no audible speech",
                    )
                )
        return DefectShare(
            eligible_turns,
            defects,
            skipped,
            defect_label,
            no_eligible_reason="No assistant turn had text or audio.",
        )

    @property
    def __name__(self):
        return "Silent Reply"
