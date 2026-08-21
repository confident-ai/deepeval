from dataclasses import dataclass, field
from typing import Dict, List

from deepeval.metrics.voice._analysis import analyze_audio, clamp_score
from deepeval.test_case import ConversationalTestCase


@dataclass
class DetectorReport:
    score: float
    critical: bool
    events: List[Dict] = field(default_factory=list)


_NON_RESPONSE_ENDINGS = (
    "bye",
    "goodbye",
    "thank you",
    "thanks",
    "that's all",
    "that is all",
)


def _response_was_owed(content: str) -> bool:
    normalized = content.strip().lower().rstrip(".!")
    return not any(
        normalized.endswith(ending) for ending in _NON_RESPONSE_ENDINGS
    )


def detect_responsiveness(test_case: ConversationalTestCase) -> DetectorReport:
    events: List[Dict] = []
    critical = False
    reprompts = 0
    turns = test_case.turns
    for index, turn in enumerate(turns):
        if turn.role != "user" or not _response_was_owed(turn.content):
            continue
        next_turn = turns[index + 1] if index + 1 < len(turns) else None
        if next_turn is None:
            critical = True
            events.append(
                {
                    "type": "agent_failed_to_respond",
                    "turn": index,
                    "critical": True,
                }
            )
            continue
        if next_turn.role == "user":
            reprompts += 1
            events.append(
                {
                    "type": "user_reprompted",
                    "turn": index + 1,
                    "critical": False,
                }
            )
            continue
        if next_turn.audio is None:
            critical = True
            events.append(
                {
                    "type": "assistant_audio_missing",
                    "turn": index + 1,
                    "critical": True,
                }
            )

    metadata = test_case.metadata or {}
    end_reason = metadata.get("end_reason") or metadata.get("endReason")
    if isinstance(end_reason, str) and end_reason.upper() in {
        "AGENT_HANGUP",
        "ERROR",
        "IDLE_TIMEOUT",
    }:
        critical = True
        events.append(
            {
                "type": "unexpected_end",
                "end_reason": end_reason,
                "critical": True,
            }
        )
    score = 0.0 if critical else clamp_score(1.0 - reprompts * 0.25)
    return DetectorReport(score=score, critical=critical, events=events)


def detect_audio_integrity(test_case: ConversationalTestCase) -> DetectorReport:
    events: List[Dict] = []
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
            events=[
                {
                    "type": "assistant_turn_missing",
                    "critical": True,
                }
            ],
        )

    for index, turn in assistant_turns:
        if turn.audio is None:
            critical = True
            events.append(
                {
                    "type": "audio_missing",
                    "turn": index,
                    "critical": True,
                }
            )
            continue
        try:
            measurements = analyze_audio(turn.audio)
        except (TypeError, ValueError) as error:
            critical = True
            events.append(
                {
                    "type": "audio_undecodable",
                    "turn": index,
                    "critical": True,
                    "reason": str(error),
                }
            )
            continue
        if measurements.ends_abruptly:
            penalty += 0.12
            events.append(
                {
                    "type": "abrupt_cutoff",
                    "turn": index,
                    "critical": False,
                }
            )
        if measurements.loop_events:
            severity = min(0.4, measurements.loop_events * 0.15)
            penalty += severity
            events.append(
                {
                    "type": "audio_loop",
                    "turn": index,
                    "count": measurements.loop_events,
                    "severity": severity,
                    "critical": measurements.loop_events >= 3,
                }
            )
            critical = critical or measurements.loop_events >= 3
        if measurements.dropout_events:
            severity = min(0.35, measurements.dropout_events * 0.08)
            penalty += severity
            events.append(
                {
                    "type": "audio_dropout",
                    "turn": index,
                    "count": measurements.dropout_events,
                    "severity": severity,
                    "critical": False,
                }
            )
        if measurements.clipping_fraction > 0.01:
            severity = min(0.35, measurements.clipping_fraction * 10.0)
            penalty += severity
            events.append(
                {
                    "type": "clipping",
                    "turn": index,
                    "fraction": measurements.clipping_fraction,
                    "severity": severity,
                    "critical": False,
                }
            )
    score = 0.0 if critical else clamp_score(1.0 - penalty)
    return DetectorReport(score=score, critical=critical, events=events)
