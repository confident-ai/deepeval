"""Shared pieces of the single-defect voice metrics. Each one scores the
share of agent replies free of its one defect; caller audio is ignored."""

from typing import Dict, List, Tuple

from deepeval.metrics.voice._analysis import AudioMeasurements, analyze_audio
from deepeval.metrics.voice.base_metric import VoiceMetricResult
from deepeval.test_case import ConversationalTestCase


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


def defect_share_result(
    eligible_turns: List[int],
    defects: List[Dict],
    skipped: List[Dict],
    defect_label: str,
) -> VoiceMetricResult:
    defective_turns = {defect["turn"] for defect in defects}
    breakdown = {
        "eligible_turns": len(eligible_turns),
        "turns_with_events": len(defective_turns),
        "events": sorted(defects, key=lambda defect: defect["turn"]),
        "skipped": skipped,
    }
    if not eligible_turns:
        return (
            None,
            "No decodable assistant audio was available to evaluate.",
            breakdown,
        )
    if not defective_turns:
        return 1.0, f"No agent reply had {defect_label}.", breakdown
    return (
        1.0 - len(defective_turns) / len(eligible_turns),
        f"{len(defective_turns)} of {len(eligible_turns)} agent replies "
        f"had {defect_label}.",
        breakdown,
    )
