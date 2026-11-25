"""Helpers for the roll-up voice metrics, which score one call over several
single-defect metrics."""

from typing import Dict, List, Optional

from deepeval.metrics.voice.utils.call_activity import caller_lines
from deepeval.metrics.voice.utils.defects import agent_reply_measurements
from deepeval.metrics.voice.utils.events import VoiceEvent
from deepeval.test_case import ConversationalTestCase

EXCHANGES = ("exchange", "exchanges")
NO_ELIGIBLE_PARTS_REASON = "None of this metric's checks had anything to judge."
# Agent sound on a recording can start a little before its turn's timestamp.
TURN_START_TOLERANCE_SECONDS = 0.5


def agent_turn_at(
    test_case: ConversationalTestCase, seconds: float
) -> Optional[int]:
    """The agent turn playing at a moment on the call: the latest one that
    started by then. None before the first one, as when the moment falls in
    turns left out of the test case."""
    starts = sorted(
        (turn.audio.start_time, index)
        for index, turn in enumerate(test_case.turns)
        if turn.role == "assistant"
        and turn.audio is not None
        and turn.audio.start_time is not None
    )
    if not starts:
        return None
    started = [index for start, index in starts if start <= seconds]
    if started:
        return started[-1]
    first_start, first_turn = starts[0]
    if first_start - seconds <= TURN_START_TOLERANCE_SECONDS:
        return first_turn
    return None


def caller_line_start_of_turn(
    test_case: ConversationalTestCase, use_speech_detection: bool
) -> Dict[int, int]:
    """Each caller turn mapped to the first turn of its caller line."""
    return {
        turn: first
        for first, last in caller_lines(test_case.turns, use_speech_detection)
        for turn in range(first, last + 1)
    }


def exchange_of(
    test_case: ConversationalTestCase,
    turn: int,
    line_start_of_turn: Dict[int, int],
) -> int:
    """The caller line a turn belongs to, by the line's first turn, or the
    turn itself when the agent speaks before the caller has said anything."""
    caller_turn = next(
        (
            index
            for index in range(turn, -1, -1)
            if test_case.turns[index].role == "user"
        ),
        None,
    )
    return turn if caller_turn is None else line_start_of_turn[caller_turn]


def agent_audio_critical_events(
    test_case: ConversationalTestCase,
) -> List[VoiceEvent]:
    """A call with no agent turns, or agent audio that cannot be decoded."""
    if not any(turn.role == "assistant" for turn in test_case.turns):
        return [VoiceEvent(type="assistant_turn_missing")]
    _, skipped = agent_reply_measurements(test_case)
    return [
        VoiceEvent(
            type="audio_undecodable", turn=entry["turn"], reason=entry["error"]
        )
        for entry in skipped
    ]
