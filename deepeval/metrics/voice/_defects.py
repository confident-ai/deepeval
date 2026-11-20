"""Shared pieces of the single-defect voice metrics. Each one scores the
share of agent replies free of its one defect. Caller audio is never judged;
the timing metrics only read when the caller's turns start and end."""

from statistics import median
from typing import Dict, List, Optional, Tuple

from deepeval.metrics.voice._analysis import (
    CONTIGUOUS_CLIP_TOLERANCE_SECONDS,
    AudioMeasurements,
    analyze_audio,
)
from deepeval.metrics.voice.base_metric import VoiceMetricResult
from deepeval.test_case import ConversationalTestCase
from deepeval.voice.timeline import AudioTimelineEntry, build_audio_timeline

# With fewer replies there is no telling which one is off.
MIN_COMPARED_REPLIES = 3
# Brief overlaps between speakers are normal; this also covers the quiet tail
# of the caller's clip.
AGENT_OVERLAP_ALLOWANCE_SECONDS = 0.5
NO_CALL_AUDIO_REASON = "The call has no audio to evaluate."
# Below this a reply carries no audible sound at all.
INAUDIBLE_MAX_DBFS = -70
INAUDIBLE_ERROR = "Reply has no audible sound to measure."


def agent_reply_measurements(
    test_case: ConversationalTestCase,
) -> Tuple[Dict[int, AudioMeasurements], List[Dict]]:
    measurements: Dict[int, AudioMeasurements] = {}
    skipped: List[Dict] = []
    for index, turn in enumerate(test_case.turns):
        if turn.role != "assistant" or turn.audio is None:
            continue
        try:
            measurements[index] = analyze_audio(turn.audio)
        except (TypeError, ValueError) as error:
            skipped.append({"turn": index, "error": str(error)})
    return measurements, skipped


# An inaudible reply fails Silent Reply, so level metrics skip it.
def audible_replies(
    measurements: Dict[int, AudioMeasurements], skipped: List[Dict]
) -> Dict[int, AudioMeasurements]:
    replies = {}
    for turn, reply in measurements.items():
        if reply.rms_dbfs > INAUDIBLE_MAX_DBFS:
            replies[turn] = reply
        else:
            skipped.append({"turn": turn, "error": INAUDIBLE_ERROR})
    return replies


def agent_glitch_events(
    test_case: ConversationalTestCase,
) -> Tuple[List[int], List[Dict], List[Dict]]:
    # audio_glitch imports this package, so a top-level import is circular.
    from deepeval.metrics.audio_glitch._detectors import detect_audio_glitches

    report = detect_audio_glitches(test_case)
    agent_turns = [
        index
        for index in report.turns_with_audio
        if test_case.turns[index].role == "assistant"
    ]
    skipped = [
        {"turn": event["turn"], "error": event["reason"]}
        for event in report.events
        if event["type"] == "audio_undecodable" and event["turn"] in agent_turns
    ]
    undecodable = {entry["turn"] for entry in skipped}
    eligible_turns = [
        index for index in agent_turns if index not in undecodable
    ]
    events = [
        event for event in report.events if event["turn"] in eligible_turns
    ]
    return eligible_turns, events, skipped


def untimed_call_reason(test_case: ConversationalTestCase) -> Optional[str]:
    if any(
        turn.audio is not None and turn.audio.start_time is None
        for turn in test_case.turns
    ):
        return (
            "Every audio turn needs a start_time to time the call, so none "
            "was inferred."
        )
    return None


def call_has_audio(test_case: ConversationalTestCase) -> bool:
    return any(turn.audio is not None for turn in test_case.turns)


def call_timeline(
    test_case: ConversationalTestCase,
) -> Tuple[List[AudioTimelineEntry], List[Dict], Optional[str]]:
    """The call's audio turns in time order, the agent replies left out because
    their audio could not be measured, or why the call cannot be timed."""
    untimed_reason = untimed_call_reason(test_case)
    if untimed_reason:
        return [], [], untimed_reason
    timeline = build_audio_timeline(test_case.turns)
    timed = {entry.turn_index for entry in timeline}
    skipped = [
        {
            "turn": index,
            "error": "Reply audio could not be decoded to measure its length.",
        }
        for index, turn in enumerate(test_case.turns)
        if turn.role == "assistant"
        and turn.audio is not None
        and index not in timed
    ]
    return timeline, skipped, None


def agent_reply_gaps(
    test_case: ConversationalTestCase,
) -> Tuple[Dict[int, float], List[Dict], Optional[str]]:
    """Seconds from the end of each caller turn to the agent reply after it;
    negative when the agent started before the caller's audio ended."""
    timeline, skipped, untimed_reason = call_timeline(test_case)
    timed = {entry.turn_index for entry in timeline}
    gaps: Dict[int, float] = {}
    for position, reply in enumerate(timeline):
        if reply.role != "assistant":
            continue
        caller_index = reply.turn_index - 1
        if (
            caller_index >= 0
            and caller_index not in timed
            and test_case.turns[caller_index].role == "user"
            and test_case.turns[caller_index].audio is not None
        ):
            skipped.append(
                {
                    "turn": reply.turn_index,
                    "error": "The caller's audio before this reply could not "
                    "be decoded to time it.",
                }
            )
            continue
        previous = timeline[position - 1] if position else None
        if previous is None or previous.role != "user":
            continue
        gap = reply.start_time - previous.end_time
        if abs(gap) <= CONTIGUOUS_CLIP_TOLERANCE_SECONDS:
            latency_ms = test_case.turns[reply.turn_index].latency_ms
            if latency_ms is None:
                skipped.append(
                    {
                        "turn": reply.turn_index,
                        "error": "Reply audio touches the caller's with no "
                        "gap and no recorded latency.",
                    }
                )
                continue
            gap = latency_ms / 1000
        gaps[reply.turn_index] = gap
    return gaps, skipped, untimed_reason


def agent_reply_delays(
    test_case: ConversationalTestCase,
) -> Tuple[Dict[int, float], List[Dict], Optional[str]]:
    """How long the agent waited before each reply. Replies that talked over
    the caller had no wait and are left to Agent Interruption."""
    gaps, skipped, untimed_reason = agent_reply_gaps(test_case)
    delays: Dict[int, float] = {}
    for turn, gap in gaps.items():
        if gap < -AGENT_OVERLAP_ALLOWANCE_SECONDS:
            skipped.append(
                {
                    "turn": turn,
                    "error": "Agent started before the caller finished.",
                }
            )
            continue
        delays[turn] = max(gap, 0.0)
    return delays, skipped, untimed_reason


def shifts_from_typical(levels: Dict[int, float]) -> Dict[int, float]:
    """Each reply's distance from the call's typical (median) reply."""
    if not levels:
        return {}
    typical = median(levels.values())
    return {turn: level - typical for turn, level in levels.items()}


def defect_share_result(
    eligible_turns: List[int],
    defects: List[Dict],
    skipped: List[Dict],
    defect_label: str,
    no_eligible_reason: str = (
        "No decodable assistant audio was available to evaluate."
    ),
    min_eligible_turns: int = 1,
    counted: Tuple[str, str] = ("agent reply", "agent replies"),
) -> VoiceMetricResult:
    defective_turns = {defect["turn"] for defect in defects}
    breakdown = {
        "eligible_turns": len(eligible_turns),
        "turns_with_events": len(defective_turns),
        "events": sorted(defects, key=lambda defect: defect["turn"]),
        "skipped": skipped,
    }
    if len(eligible_turns) < min_eligible_turns:
        return None, no_eligible_reason, breakdown
    if not defective_turns:
        return 1.0, f"No {counted[0]} had {defect_label}.", breakdown
    return (
        1.0 - len(defective_turns) / len(eligible_turns),
        f"{len(defective_turns)} of {len(eligible_turns)} {counted[1]} "
        f"had {defect_label}.",
        breakdown,
    )
