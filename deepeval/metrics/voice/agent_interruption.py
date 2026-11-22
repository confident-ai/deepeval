from typing import Callable, Dict, Optional

from deepeval.metrics.voice._call_activity import (
    EVENT_KEYS,
    AgentStart,
    agent_starts,
    call_activity,
)
from deepeval.metrics.voice._defects import defect_share_result
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase

FALSE_START_MAX_SECONDS = 1.5
TALK_OVER_MIN_SECONDS = 1.0


def is_false_start(
    start: AgentStart, metric: "AgentInterruptionMetric"
) -> bool:
    """The agent starts speaking AND (the caller is already talking OR starts
    within the caller-start window) AND the agent stops within 1.5 s; not
    counted when the caller's pause ran past the caller-silence limit."""
    caller_collides = start.caller_talking or (
        start.caller_starts_after is not None
        and start.caller_starts_after <= metric.caller_start_window_ms / 1000
    )
    caller_was_waiting = (
        start.caller_pause_seconds is not None
        and start.caller_pause_seconds > metric.max_caller_silence_seconds
    )
    return (
        caller_collides
        and start.stops_at - start.at <= FALSE_START_MAX_SECONDS
        and not caller_was_waiting
    )


def is_talk_over(start: AgentStart, metric: "AgentInterruptionMetric") -> bool:
    """The agent starts speaking AND the caller is already talking AND both
    keep talking for 1 s or more."""
    return (
        start.caller_talking and start.overlap_seconds >= TALK_OVER_MIN_SECONDS
    )


# Interruption == False start OR Talk-over. Add a condition by adding it here.
INTERRUPTION_CONDITIONS: Dict[
    str, Callable[[AgentStart, "AgentInterruptionMetric"], bool]
] = {
    "false_start": is_false_start,
    "talk_over": is_talk_over,
}


class AgentInterruptionMetric(BaseVoiceMetric):
    # On human-labelled phone calls, reviewers excused an agent that spoke
    # after about 4 s of caller silence.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        caller_start_window_ms: int = 500,
        max_caller_silence_seconds: int = 4,
        flaky: bool = False,
    ):
        if caller_start_window_ms <= 0:
            raise ValueError("caller_start_window_ms must be greater than 0.")
        if max_caller_silence_seconds <= 0:
            raise ValueError(
                "max_caller_silence_seconds must be greater than 0."
            )
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.caller_start_window_ms = caller_start_window_ms
        self.max_caller_silence_seconds = max_caller_silence_seconds

    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        activity = call_activity(test_case)
        event_key = EVENT_KEYS[activity.source]
        eligible = []
        defects = []
        # The opening line interrupts no one.
        for start in agent_starts(activity)[1:]:
            key = start.turn if event_key == "turn" else round(start.at, 2)
            eligible.append(key)
            fired = [
                name
                for name, condition in INTERRUPTION_CONDITIONS.items()
                if condition(start, self)
            ]
            if fired:
                defects.append(
                    {
                        "type": fired[0],
                        event_key: key,
                        "agent_seconds": round(start.stops_at - start.at, 2),
                        "overlap_seconds": round(start.overlap_seconds, 2),
                    }
                )
        return defect_share_result(
            eligible,
            defects,
            list(activity.skipped),
            "the agent interrupting the caller",
            no_eligible_reason=activity.untimed_reason
            or "The agent never started speaking after its opening line.",
            event_key=event_key,
            timing_source=activity.source,
        )

    @property
    def __name__(self):
        return "Agent Interruption"
