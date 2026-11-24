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
from deepeval.metrics.voice.utils.decision_check import turn_clip
from deepeval.metrics.voice.utils.detectors import response_was_owed
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import RANGE, VoiceEvent


class RepromptMetric(BaseDefectMetric):
    event_question = "Is the caller repeating what they said, or checking whether anyone is still there?"
    event_speaker = "user"

    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        audio_decision_model: Optional[DeepEvalBaseAudioDecisionModel] = None,
        flaky: bool = False,
    ):
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self._use_audio_decision_model(audio_decision_model)

    def _event_clip(
        self, test_case: ConversationalTestCase, event: VoiceEvent
    ) -> Optional[bytes]:
        return turn_clip(test_case, event.get("reprompt_turn"))

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
                silence = silence_after_line(turns, last)
                wait = silence.get("duration_seconds")
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
