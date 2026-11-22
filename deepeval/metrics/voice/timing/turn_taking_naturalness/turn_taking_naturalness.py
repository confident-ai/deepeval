import math
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

from deepeval.metrics.voice.utils.analysis import (
    CONTIGUOUS_CLIP_TOLERANCE_SECONDS,
)
from deepeval.metrics.voice.utils.call_activity import (
    SPEAKER_PAUSE_SECONDS,
    CallActivity,
    call_recording_activity,
    stretches,
)
from deepeval.metrics.voice.utils.defects import CALL_RECORDING, TURN_TIMESTAMPS
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase
from deepeval.voice.timeline import build_audio_timeline

# How evenly the agent waits before replying: scored once there are enough
# caller-to-agent gaps to compare, and weighted so it adjusts rather than
# dominates the transition scores.
RESPONSE_DELAY_MIN_COUNT = 3
RESPONSE_DELAY_SPREAD_TAU = 2.0
RESPONSE_DELAY_WEIGHT = 0.2
# Each transition scores exp(-seconds / tau): silence decays slowly, the
# caller barging in a little faster, the agent talking over the caller fast.
GAP_TAU = 2.5
USER_BARGE_IN_TAU = 2.0
AGENT_OVERLAP_TAU = 0.6
# On a recording, shorter overlaps are speakers touching, and a caller sound
# this short over the agent is an "mm-hm", not a hand-over.
MIN_OVERLAP_SECONDS = 0.3
BACKCHANNEL_MAX_SECONDS = 1.0


def _recording_transitions(
    recording: CallActivity,
) -> Tuple[List[Dict], List[float], List[float]]:
    """Hand-over == Gap OR Overlap. Gap: a speaker starts while the other is
    silent and the other spoke last. Overlap: a speaker starts while the
    other is talking."""
    floors = {
        "assistant": stretches(recording.agent, SPEAKER_PAUSE_SECONDS),
        "user": stretches(recording.caller, SPEAKER_PAUSE_SECONDS),
    }
    opening = floors["assistant"][0] if floors["assistant"] else None
    transitions: List[Dict] = []
    scores: List[float] = []
    response_delay_seconds: List[float] = []
    starts = sorted(
        (
            (role, stretch)
            for role, spans in floors.items()
            for stretch in spans
        ),
        key=lambda item: item[1].start,
    )
    for role, stretch in starts:
        if stretch is opening:
            continue
        other_role = "user" if role == "assistant" else "assistant"
        other = floors[other_role]
        talking = next(
            (span for span in other if span.start < stretch.start < span.end),
            None,
        )
        if talking is not None:
            overlap = min(stretch.end, talking.end) - stretch.start
            if overlap < MIN_OVERLAP_SECONDS or (
                role == "user"
                and stretch.end - stretch.start < BACKCHANNEL_MAX_SECONDS
            ):
                continue
            if role == "user":
                transition_score = math.exp(-overlap / USER_BARGE_IN_TAU)
                kind = "user_barge_in"
            else:
                transition_score = math.exp(-overlap / AGENT_OVERLAP_TAU)
                kind = "agent_overlap"
            gap_seconds = -overlap
        else:
            other_end = max(
                (span.end for span in other if span.end <= stretch.start),
                default=None,
            )
            own_end = max(
                (
                    span.end
                    for span in floors[role]
                    if span.end <= stretch.start
                ),
                default=None,
            )
            if other_end is None or (
                own_end is not None and own_end > other_end
            ):
                continue
            gap_seconds = stretch.start - other_end
            transition_score = math.exp(-gap_seconds / GAP_TAU)
            kind = "gap"
            if role == "assistant":
                response_delay_seconds.append(gap_seconds)
        scores.append(transition_score)
        transitions.append(
            {
                "at_seconds": round(stretch.start, 2),
                "from": other_role,
                "to": role,
                "kind": kind,
                "gap_seconds": round(gap_seconds, 2),
                "score": transition_score,
            }
        )
    return transitions, scores, response_delay_seconds


def _naturalness_result(
    transitions: List[Dict],
    scores: List[float],
    response_delay_seconds: List[float],
    unmeasured_transitions: List[Dict],
) -> VoiceMetricResult:
    score = mean(scores)
    response_delays: Dict = {
        "count": len(response_delay_seconds),
        "mean_seconds": (
            mean(response_delay_seconds) if response_delay_seconds else None
        ),
        "spread_seconds": None,
        "score": None,
    }
    reason = (
        f"Turn-taking naturalness was {score:.2f} across "
        f"{len(scores)} speaker transition(s); timing is scored "
        "continuously rather than against a latency cutoff."
    )
    if len(response_delay_seconds) >= RESPONSE_DELAY_MIN_COUNT:
        spread = pstdev(response_delay_seconds)
        consistency_score = math.exp(-spread / RESPONSE_DELAY_SPREAD_TAU)
        response_delays["spread_seconds"] = spread
        response_delays["score"] = consistency_score
        score = (
            1 - RESPONSE_DELAY_WEIGHT
        ) * score + RESPONSE_DELAY_WEIGHT * consistency_score
        reason = (
            f"Turn-taking naturalness was {score:.2f} across "
            f"{len(scores)} speaker transition(s); the agent's response "
            f"delay varied by ±{spread:.2f}s across "
            f"{len(response_delay_seconds)} replies."
        )
    if unmeasured_transitions:
        reason += (
            f" {len(unmeasured_transitions)} transition(s) were left out "
            "because their audio clips touch with no measurable gap."
        )
    return (
        score,
        reason,
        {
            "transitions": transitions,
            "unmeasured_transitions": unmeasured_transitions,
            "response_delays": response_delays,
        },
    )


class TurnTakingNaturalnessMetric(BaseVoiceMetric):
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

    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        recording = call_recording_activity(
            test_case, self.use_speech_detection
        )
        if recording is None:
            score, reason, breakdown = self._evaluate_turns(test_case)
            return (
                score,
                reason,
                {**breakdown, "timing_source": TURN_TIMESTAMPS},
            )
        transitions, scores, response_delay_seconds = _recording_transitions(
            recording
        )
        if not scores:
            return (
                None,
                "No speaker hand-over was found on the call recording.",
                {"transitions": [], "timing_source": CALL_RECORDING},
            )
        score, reason, breakdown = _naturalness_result(
            transitions, scores, response_delay_seconds, []
        )
        return score, reason, {**breakdown, "timing_source": CALL_RECORDING}

    def _evaluate_turns(
        self, test_case: ConversationalTestCase
    ) -> VoiceMetricResult:
        audio_turns = [
            turn for turn in test_case.turns if turn.audio is not None
        ]
        if any(turn.audio.start_time is None for turn in audio_turns):
            return (
                None,
                "Turn-taking naturalness requires start_time on every audio "
                "turn so silence and overlap are not inferred.",
                {"missing_start_times": True},
            )
        timeline = build_audio_timeline(test_case.turns)
        if len(timeline) < 2:
            return (
                None,
                "At least two timestamped audio turns are required to "
                "evaluate turn-taking.",
                {"timeline_entries": len(timeline)},
            )

        transitions: List[Dict] = []
        unmeasured_transitions: List[Dict] = []
        scores: List[float] = []
        response_delay_seconds: List[float] = []
        for current, following in zip(timeline, timeline[1:]):
            if current.role == following.role:
                continue
            gap_seconds = following.start_time - current.end_time
            gap_source = "audio"
            if abs(gap_seconds) <= CONTIGUOUS_CLIP_TOLERANCE_SECONDS:
                latency_ms = test_case.turns[following.turn_index].latency_ms
                if current.role != "user" or latency_ms is None:
                    unmeasured_transitions.append(
                        {
                            "from_turn": current.turn_index,
                            "to_turn": following.turn_index,
                        }
                    )
                    continue
                gap_seconds = latency_ms / 1000
                gap_source = "latency"
            if gap_seconds >= 0:
                # Smoothly decreases as silence grows; there is no hard cap.
                transition_score = math.exp(-gap_seconds / GAP_TAU)
                kind = "gap"
                if current.role == "user":
                    response_delay_seconds.append(gap_seconds)
            elif current.role == "assistant" and following.role == "user":
                # User barge-in can be natural; only very long overlap is poor.
                transition_score = math.exp(
                    -abs(gap_seconds) / USER_BARGE_IN_TAU
                )
                kind = "user_barge_in"
            else:
                # Agent-on-user overlap is more disruptive.
                transition_score = math.exp(
                    -abs(gap_seconds) / AGENT_OVERLAP_TAU
                )
                kind = "agent_overlap"
            scores.append(transition_score)
            transitions.append(
                {
                    "from_turn": current.turn_index,
                    "to_turn": following.turn_index,
                    "kind": kind,
                    "gap_seconds": gap_seconds,
                    "gap_source": gap_source,
                    "score": transition_score,
                }
            )
        if not scores:
            if unmeasured_transitions:
                return (
                    None,
                    "No speaker transition had a measurable gap: every turn's "
                    "audio ends exactly where the next one starts and no "
                    "response latency was provided.",
                    {
                        "transitions": transitions,
                        "unmeasured_transitions": unmeasured_transitions,
                    },
                )
            return (
                None,
                "No cross-speaker transitions were available to evaluate.",
                {"transitions": transitions},
            )
        return _naturalness_result(
            transitions, scores, response_delay_seconds, unmeasured_transitions
        )

    @property
    def __name__(self):
        return "Turn-Taking Naturalness"
