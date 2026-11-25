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

# A caller speaking again sooner is pausing mid-thought, not kept waiting:
# it is the wait Response Latency's default counts as slow.
REPROMPT_MIN_WAIT_SECONDS = 4.0


class RepromptMetric(BaseDefectMetric):
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
        for first, last in caller_lines(turns, self.use_speech_detection):
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
                silence = silence_after_line(
                    turns, last, self.use_speech_detection
                )
                wait = silence.get("duration_seconds")
                repeat_is_timed = (
                    next_turn.audio is not None
                    and next_turn.audio.start_time is not None
                )
                if (
                    repeat_is_timed
                    and wait is not None
                    and wait < REPROMPT_MIN_WAIT_SECONDS
                ):
                    continue
                defects.append(
                    VoiceEvent(
                        type="user_reprompted",
                        turn=first,
                        reprompt_turn=last + 1,
                        mark=RANGE,
                        description=(
                            "Caller repeated themselves"
                            if wait is None
                            else f"Caller repeated themselves after {wait:.1f}s"
                        ),
                        **silence,
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
