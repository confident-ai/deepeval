from dataclasses import dataclass, field
from typing import List

from deepeval.metrics.voice.utils.analysis import analyze_audio, clamp_score
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent, placed


@dataclass
class DetectorReport:
    score: float
    critical: bool
    events: List[VoiceEvent] = field(default_factory=list)


# Platform end reasons that mean the call did not end normally.
UNEXPECTED_END_REASONS = {"AGENT_HANGUP", "ERROR", "IDLE_TIMEOUT"}

_NON_RESPONSE_ENDINGS = (
    "bye",
    "goodbye",
    "thank you",
    "thanks",
    "that's all",
    "that is all",
)


def response_was_owed(content: str) -> bool:
    normalized = content.strip().lower().rstrip(".!")
    return not any(
        normalized.endswith(ending) for ending in _NON_RESPONSE_ENDINGS
    )


def detect_responsiveness(test_case: ConversationalTestCase) -> DetectorReport:
    events: List[VoiceEvent] = []
    critical = False
    reprompts = 0
    turns = test_case.turns
    for index, turn in enumerate(turns):
        if turn.role != "user" or not response_was_owed(turn.content):
            continue
        next_turn = turns[index + 1] if index + 1 < len(turns) else None
        if next_turn is None:
            critical = True
            events.append(
                VoiceEvent(
                    type="agent_failed_to_respond", turn=index, critical=True
                )
            )
            continue
        if next_turn.role == "user":
            reprompts += 1
            events.append(
                VoiceEvent(
                    type="user_reprompted", turn=index + 1, critical=False
                )
            )
            continue
        if next_turn.audio is None:
            critical = True
            events.append(
                VoiceEvent(
                    type="assistant_audio_missing",
                    turn=index + 1,
                    critical=True,
                )
            )

    metadata = test_case.metadata or {}
    end_reason = metadata.get("end_reason") or metadata.get("endReason")
    if (
        isinstance(end_reason, str)
        and end_reason.upper() in UNEXPECTED_END_REASONS
    ):
        critical = True
        events.append(
            VoiceEvent(
                type="unexpected_end", end_reason=end_reason, critical=True
            )
        )
    score = 0.0 if critical else clamp_score(1.0 - reprompts * 0.25)
    return DetectorReport(score=score, critical=critical, events=events)


def detect_audio_integrity(test_case: ConversationalTestCase) -> DetectorReport:
    events: List[VoiceEvent] = []
    critical = False
    penalty = 0.0
    assistant_turns = [
        (index, turn)
        for index, turn in enumerate(test_case.turns)
        if turn.role == "assistant"
    ]
    if not assistant_turns:
        return DetectorReport(
            score=0.0,
            critical=True,
            events=[VoiceEvent(type="assistant_turn_missing", critical=True)],
        )

    for index, turn in assistant_turns:
        if turn.audio is None:
            critical = True
            events.append(
                VoiceEvent(type="audio_missing", turn=index, critical=True)
            )
            continue
        try:
            measurements = analyze_audio(turn.audio)
        except (TypeError, ValueError) as error:
            critical = True
            events.append(
                VoiceEvent(
                    type="audio_undecodable",
                    turn=index,
                    critical=True,
                    reason=str(error),
                )
            )
            continue
        if measurements.ends_abruptly:
            penalty += 0.12
            events.append(
                VoiceEvent(type="abrupt_cutoff", turn=index, critical=False)
            )
        if measurements.loop_events:
            severity = min(0.4, measurements.loop_events * 0.15)
            penalty += severity
            for offset, duration in measurements.loop_spans:
                events.append(
                    VoiceEvent(
                        type="audio_loop",
                        turn=index,
                        critical=measurements.loop_events >= 3,
                        **placed(turn.audio, offset, duration),
                    )
                )
            critical = critical or measurements.loop_events >= 3
        if measurements.dropout_events:
            severity = min(0.35, measurements.dropout_events * 0.08)
            penalty += severity
            for offset, duration in measurements.dropout_spans:
                events.append(
                    VoiceEvent(
                        type="audio_dropout",
                        turn=index,
                        critical=False,
                        **placed(turn.audio, offset, duration),
                    )
                )
        if measurements.clipping_fraction > 0.01:
            severity = min(0.35, measurements.clipping_fraction * 10.0)
            penalty += severity
            events.append(
                VoiceEvent(
                    type="clipping",
                    turn=index,
                    fraction=measurements.clipping_fraction,
                    severity=severity,
                    critical=False,
                )
            )
    score = 0.0 if critical else clamp_score(1.0 - penalty)
    return DetectorReport(score=score, critical=critical, events=events)
