"""Sub-scores of turn-taking: one named behaviour each, from 0 to 1.

Each returns `None` when the conversation never tested that behaviour, so a
call without a pause says nothing about pauses.

Caller events are read from `Turn.metadata["events"]` on user turns, as
`{"type", "offset_seconds", "seconds"}` with offsets from the start of that
turn's audio. Backchannels are read from `Turn.metadata["backchannels"]` on
assistant turns.
"""

import math
from statistics import mean, median
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.voice.timeline import AudioTimelineEntry

# Longest the agent may stay silent before a reply that used a tool.
TOOL_SILENCE_SECONDS = 3.0
# How long after a backchannel the agent must still be talking.
BACKCHANNEL_CARRY_ON_SECONDS = 1.0
# Check-ins allowed while the caller is away or silent.
MAX_CHECK_INS = 2

SubScore = Tuple[Optional[float], Dict]
_ENDS_SENTENCE = (".", "!", "?", "…")


def _events(turn: Turn, kind: str) -> List[Dict]:
    return [
        event
        for event in (turn.metadata or {}).get("events", [])
        if event.get("type") == kind
    ]


def _share(passed: List[bool]) -> Optional[float]:
    return sum(passed) / len(passed) if passed else None


def _agent_starts(
    timeline: Sequence[AudioTimelineEntry], start: float, end: float
) -> int:
    return sum(
        1
        for entry in timeline
        if entry.role == "assistant" and start <= entry.start_time < end
    )


def _hung_up(test_case: ConversationalTestCase) -> bool:
    return (test_case.metadata or {}).get(
        "Stop reason"
    ) == "The agent ended the call"


def reply_speed(test_case, timeline) -> SubScore:
    gaps = [
        following.start_time - current.end_time
        for current, following in zip(timeline, timeline[1:])
        if current.role == "user"
        and following.role == "assistant"
        and following.start_time >= current.end_time
    ]
    if not gaps:
        return None, {}
    typical = median(gaps)
    slowest = sorted(gaps)[max(0, math.ceil(len(gaps) * 0.9) - 1)]
    return math.exp(-typical / 2.5), {
        "p50_seconds": typical,
        "p90_seconds": slowest,
        "replies": len(gaps),
    }


def premature_response(test_case, timeline) -> SubScore:
    overlaps = [
        current.end_time - following.start_time
        for current, following in zip(timeline, timeline[1:])
        if current.role == "user"
        and following.role == "assistant"
        and following.start_time < current.end_time
    ]
    handoffs = sum(
        1
        for current, following in zip(timeline, timeline[1:])
        if current.role == "user" and following.role == "assistant"
    )
    if not handoffs:
        return None, {}
    score = 1 - len(overlaps) / handoffs
    return score, {"early_replies": len(overlaps), "handoffs": handoffs}


def overlapping_speech(test_case, timeline) -> SubScore:
    overlaps = [
        current.end_time - following.start_time
        for current, following in zip(timeline, timeline[1:])
        if current.role == "assistant"
        and following.role == "user"
        and following.start_time < current.end_time
    ]
    if not overlaps:
        return None, {}
    return mean(math.exp(-overlap / 2.0) for overlap in overlaps), {
        "cut_ins": len(overlaps),
        "seconds_to_stop": overlaps,
    }


def mid_utterance_pauses(test_case, timeline) -> SubScore:
    waited: List[bool] = []
    for entry in timeline:
        if entry.role != "user":
            continue
        for pause in _events(
            test_case.turns[entry.turn_index], "MidSentencePause"
        ):
            start = entry.start_time + pause["offset_seconds"]
            waited.append(
                _agent_starts(timeline, start, start + pause["seconds"]) == 0
            )
    return _share(waited), {
        "pauses": len(waited),
        "talked_into": waited.count(False),
    }


def backchannel_vs_barge_in(test_case, timeline) -> SubScore:
    kept_going: List[bool] = []
    for entry in timeline:
        turn = test_case.turns[entry.turn_index]
        if entry.role != "assistant":
            continue
        for backchannel in (turn.metadata or {}).get("backchannels", []):
            carried_on = (
                entry.end_time
                - (entry.start_time + backchannel["offset_seconds"])
                >= BACKCHANNEL_CARRY_ON_SECONDS
            )
            finished = turn.content.rstrip().endswith(_ENDS_SENTENCE)
            kept_going.append(not turn.interrupted and (carried_on or finished))
    return _share(kept_going), {
        "backchannels": len(kept_going),
        "stopped_for": kept_going.count(False),
    }


def echo(test_case, timeline) -> SubScore:
    """Agent replies cut short with no caller speech to explain it."""
    replies = [entry for entry in timeline if entry.role == "assistant"]
    if not replies:
        return None, {}
    callers = [entry for entry in timeline if entry.role == "user"]
    self_interrupted = 0
    for reply in replies:
        turn = test_case.turns[reply.turn_index]
        caller_spoke = any(
            caller.start_time < reply.end_time
            and caller.end_time > reply.start_time
            for caller in callers
        )
        if turn.interrupted and not caller_spoke:
            self_interrupted += 1
    return 1 - self_interrupted / len(replies), {
        "replies": len(replies),
        "self_interrupted": self_interrupted,
    }


def tool_latency(test_case, timeline) -> SubScore:
    silences: List[float] = []
    for current, following in zip(timeline, timeline[1:]):
        turn = test_case.turns[following.turn_index]
        if (
            current.role == "user"
            and following.role == "assistant"
            and turn.tools_called
        ):
            silences.append(max(0.0, following.start_time - current.end_time))
    filled = [silence <= TOOL_SILENCE_SECONDS for silence in silences]
    return _share(filled), {
        "tool_replies": len(silences),
        "silence_seconds": silences,
        "limit_seconds": TOOL_SILENCE_SECONDS,
    }


def _waits_through(kind: str) -> Callable:
    def score(test_case, timeline) -> SubScore:
        results: List[float] = []
        check_ins: List[int] = []
        for entry in timeline:
            if entry.role != "user":
                continue
            for event in _events(test_case.turns[entry.turn_index], kind):
                start = entry.start_time + event["offset_seconds"]
                spoke = _agent_starts(timeline, start, start + event["seconds"])
                check_ins.append(spoke)
                last = entry.turn_index == len(test_case.turns) - 1
                if last and _hung_up(test_case):
                    results.append(0.0)
                else:
                    results.append(1.0 if spoke <= MAX_CHECK_INS else 0.5)
        if not results:
            return None, {}
        return mean(results), {"waits": len(results), "check_ins": check_ins}

    return score


SUB_SCORES: Dict[str, Callable] = {
    "reply_speed": reply_speed,
    "premature_response": premature_response,
    "overlapping_speech": overlapping_speech,
    "mid_utterance_pauses": mid_utterance_pauses,
    "backchannel_vs_barge_in": backchannel_vs_barge_in,
    "echo": echo,
    "tool_latency": tool_latency,
    "waiting_through_hold": _waits_through("Hold"),
    "dead_air": _waits_through("DeadAir"),
}
