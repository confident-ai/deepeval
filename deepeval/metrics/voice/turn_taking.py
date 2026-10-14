import math
from statistics import mean, pstdev
from typing import Dict, List

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


class TurnTakingNaturalnessMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
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
        scores: List[float] = []
        response_delay_seconds: List[float] = []
        for current, following in zip(timeline, timeline[1:]):
            if current.role == following.role:
                continue
            gap_seconds = following.start_time - current.end_time
            if gap_seconds >= 0:
                # Smoothly decreases as silence grows; there is no hard cap.
                transition_score = math.exp(-gap_seconds / 2.5)
                kind = "gap"
                if current.role == "user":
                    response_delay_seconds.append(gap_seconds)
            elif current.role == "assistant" and following.role == "user":
                # User barge-in can be natural; only very long overlap is poor.
                transition_score = math.exp(-abs(gap_seconds) / 2.0)
                kind = "user_barge_in"
            else:
                # Agent-on-user overlap is more disruptive.
                transition_score = math.exp(-abs(gap_seconds) / 0.6)
                kind = "agent_overlap"
            scores.append(transition_score)
            transitions.append(
                {
                    "from_turn": current.turn_index,
                    "to_turn": following.turn_index,
                    "kind": kind,
                    "gap_seconds": gap_seconds,
                    "score": transition_score,
                }
            )
        if not scores:
            return (
                None,
                "No cross-speaker transitions were available to evaluate.",
                {"transitions": transitions},
            )
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
        return (
            score,
            reason,
            {"transitions": transitions, "response_delays": response_delays},
        )

    @property
    def __name__(self):
        return "Turn-Taking Naturalness"
