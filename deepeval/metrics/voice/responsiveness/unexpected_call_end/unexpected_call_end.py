import re
from typing import Optional

from deepeval.metrics.voice.utils.defects import (
    NO_CALL_AUDIO_REASON,
    call_has_audio,
    DefectShare,
)
from deepeval.metrics.voice.utils.detectors import (
    UNEXPECTED_END_REASONS,
    response_was_owed,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.call_activity import call_end
from deepeval.metrics.voice.utils.events import MOMENT, VoiceEvent, on_call

# Written to the test case's metadata by ConversationSimulator.
STOP_REASON_KEY = "Stop reason"
AGENT_ENDED_CALL_STOP_REASON = "The agent ended the call"
AGENT_HANGUP_END_REASON = "AGENT_HANGUP"
FAREWELL = re.compile(r"\b(bye|goodbye)\b")
CALLS = ("call", "calls")


def _call_wrapped_up(test_case: ConversationalTestCase) -> bool:
    last_caller = next(
        (turn for turn in reversed(test_case.turns) if turn.role == "user"),
        None,
    )
    last_agent = next(
        (
            turn
            for turn in reversed(test_case.turns)
            if turn.role == "assistant"
        ),
        None,
    )
    return (
        last_caller is not None and not response_was_owed(last_caller.content)
    ) or (
        last_agent is not None
        and FAREWELL.search(last_agent.content.lower()) is not None
    )


def _unexpected_end_reason(test_case: ConversationalTestCase) -> Optional[str]:
    metadata = test_case.metadata or {}
    end_reason = metadata.get("end_reason") or metadata.get("endReason")
    end_reason = end_reason if isinstance(end_reason, str) else None
    if end_reason and end_reason.upper() in UNEXPECTED_END_REASONS - {
        AGENT_HANGUP_END_REASON
    }:
        return end_reason
    if _call_wrapped_up(test_case):
        return None
    if metadata.get(STOP_REASON_KEY) == AGENT_ENDED_CALL_STOP_REASON:
        return "agent_ended_call"
    if end_reason and end_reason.upper() == AGENT_HANGUP_END_REASON:
        return end_reason
    return None


class UnexpectedCallEndMetric(BaseDefectMetric):
    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        defect_label = "an unexpected end"
        if not call_has_audio(test_case):
            return DefectShare(
                [],
                [],
                [],
                defect_label,
                no_eligible_reason=NO_CALL_AUDIO_REASON,
                counted=CALLS,
            )
        last_turn = len(test_case.turns) - 1
        end_reason = _unexpected_end_reason(test_case)
        ended_at = call_end(test_case.turns)
        defects = (
            [
                VoiceEvent(
                    type="unexpected_end",
                    turn=last_turn,
                    end_reason=end_reason,
                    mark=MOMENT,
                    description=f"Call ended unexpectedly ({end_reason})",
                    **({} if ended_at is None else on_call(ended_at, 0.0)),
                )
            ]
            if end_reason
            else []
        )
        return DefectShare(
            [last_turn], defects, [], defect_label, counted=CALLS
        )

    @property
    def __name__(self):
        return "Unexpected Call End"
