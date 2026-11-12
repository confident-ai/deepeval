from typing import Optional

from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase
from deepeval.voice.timeline import build_audio_timeline


class SimulationEfficiencyMetric(BaseVoiceMetric):
    """How directly the agent got a simulation graph conversation to its
    ending.

    The score is the fewest user turns the graph allows divided by the user
    turns the conversation took, so an agent that needed no corrections or
    repeats scores 1.
    """

    def __init__(
        self,
        expected_tool_calls: Optional[int] = None,
        threshold: Optional[float] = 0.7,
        **kwargs,
    ):
        super().__init__(threshold=threshold, **kwargs)
        self.expected_tool_calls = expected_tool_calls

    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        metadata = test_case.metadata or {}
        shortest = metadata.get("Shortest path")
        if "Simulation path" not in metadata or not shortest:
            return (
                None,
                "Efficiency needs a simulation graph with at least one ending "
                "that is not a failure.",
                {},
            )
        user_turns = sum(
            1
            for turn in test_case.turns
            if turn.role == "user" and turn.content
        )
        if not user_turns:
            return None, "The simulated user never spoke.", {}
        score = min(1.0, shortest / user_turns)
        tool_calls = sum(
            len(turn.tools_called or [])
            for turn in test_case.turns
            if turn.role == "assistant"
        )
        timeline = build_audio_timeline(test_case.turns)
        breakdown = {
            "user_turns": user_turns,
            "shortest_path": shortest,
            "extra_user_turns": max(0, user_turns - shortest),
            "tool_calls": tool_calls,
            "extra_tool_calls": (
                max(0, tool_calls - self.expected_tool_calls)
                if self.expected_tool_calls is not None
                else None
            ),
            "call_seconds": (
                max(entry.end_time for entry in timeline) if timeline else None
            ),
        }
        reason = (
            f"The caller spoke {user_turns} times; the shortest route through "
            f"the graph takes {shortest}."
        )
        return score, reason, breakdown

    @property
    def __name__(self):
        return "Simulation Efficiency"
