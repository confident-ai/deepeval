from deepeval.metrics.voice.utils.defects import (
    NO_CALL_AUDIO_REASON,
    agent_reply_measurements,
    call_has_audio,
    defect_share_result,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent


class SilentReplyMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        defect_label = "a reply the caller could not hear"
        if not call_has_audio(test_case):
            return defect_share_result(
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
                or not turn.content.strip()
                or index in undecodable
            ):
                continue
            eligible_turns.append(index)
            if turn.audio is None:
                defects.append(VoiceEvent(type="no_audio", turn=index))
            elif measurements[index].silence_fraction >= 1.0:
                defects.append(VoiceEvent(type="no_speech", turn=index))
        return defect_share_result(
            eligible_turns,
            defects,
            skipped,
            defect_label,
            no_eligible_reason="No assistant turn had text to be spoken.",
        )

    @property
    def __name__(self):
        return "Silent Reply"
