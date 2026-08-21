import math
from statistics import mean
from typing import Dict, List

from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase
from deepeval.voice.timeline import build_audio_timeline


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
        for current, following in zip(timeline, timeline[1:]):
            if current.role == following.role:
                continue
            gap_seconds = following.start_time - current.end_time
            if gap_seconds >= 0:
                # Smoothly decreases as silence grows; there is no hard cap.
                transition_score = math.exp(-gap_seconds / 2.5)
                kind = "gap"
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
        return (
            score,
            f"Turn-taking naturalness was {score:.2f} across "
            f"{len(scores)} speaker transition(s); timing is scored "
            "continuously rather than against a latency cutoff.",
            {"transitions": transitions},
        )

    @property
    def __name__(self):
        return "Turn-Taking Naturalness"
