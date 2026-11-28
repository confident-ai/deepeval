from typing import Optional

from deepeval.metrics.voice.utils.call_activity import (
    caller_lines,
    silence_after_line,
)
from deepeval.metrics.voice.utils.defects import (
    CALLER_LINES,
    NO_CALL_AUDIO_REASON,
    call_has_audio,
    DefectShare,
)
from deepeval.metrics.voice.utils.detectors import response_was_owed
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import RANGE, VoiceEvent


class FailedResponseMetric(BaseDefectMetric):
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        use_speech_detection: bool = True,
        flaky: bool = False,
    ):
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.use_speech_detection = use_speech_detection

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        defect_label = "no reply from the agent"
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
        for first, last in caller_lines(turns, self.use_speech_detection):
            if not response_was_owed(turns[last].content):
                continue
            eligible_turns.append(first)
            reply_turn = next(
                (
                    later
                    for later in range(last + 1, len(turns))
                    if turns[later].role == "assistant"
                ),
                None,
            )
            if reply_turn is None:
                defects.append(
                    VoiceEvent(
                        type="call_ended_without_reply",
                        turn=first,
                        mark=RANGE,
                        description="Call ended without a reply",
                        **silence_after_line(
                            turns, last, self.use_speech_detection
                        ),
                    )
                )
            elif (
                not turns[reply_turn].content.strip()
                and turns[reply_turn].audio is None
            ):
                defects.append(
                    VoiceEvent(
                        type="empty_reply",
                        turn=first,
                        reply_turn=reply_turn,
                        mark=RANGE,
                        description="Agent's reply was empty",
                        **silence_after_line(
                            turns, last, self.use_speech_detection
                        ),
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
        return "Failed Response"
