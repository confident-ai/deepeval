from typing import Callable, Dict, Optional

from deepeval.metrics.voice.utils.call_activity import (
    COLLISION_SECONDS,
    EVENT_KEYS,
    FALSE_START_MAX_SECONDS,
    TALK_OVER_MIN_SECONDS,
    AgentStart,
    agent_starts,
    is_opening_line,
    call_activity,
    interruption_place,
)
from deepeval.metrics.voice.utils.defects import DefectShare
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.config.voice_eval_mode import VoiceEvalModeName
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent


def is_false_start(
    start: AgentStart, metric: "AgentInterruptionMetric"
) -> bool:
    """The agent starts speaking AND (the caller is already talking OR starts
    within 0.5 s) AND the agent stops within 1.5 s; not counted when the
    caller's pause ran past the caller-silence limit."""
    caller_collides = start.caller_talking or (
        start.caller_starts_after is not None
        and start.caller_starts_after <= COLLISION_SECONDS
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

AGENT_DESCRIPTIONS = {
    "false_start": "Agent started over the caller, then stopped",
    "talk_over": "Agent talked over the caller for {overlap:.1f}s",
}


class AgentInterruptionMetric(BaseDefectMetric):
    event_question = "Is a person speaking in this clip, rather than only background noise, breathing or silence?"
    event_speaker = "user"

    # On human-labelled phone calls, reviewers excused an agent that spoke
    # after about 4 s of caller silence.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        max_caller_silence_seconds: int = 4,
        use_speech_detection: bool = True,
        audio_decision_model: Optional[DeepEvalBaseAudioDecisionModel] = None,
        eval_mode: Optional[VoiceEvalModeName] = None,
        flaky: bool = False,
    ):
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
        self._use_audio_decision_model(audio_decision_model, eval_mode)
        self.max_caller_silence_seconds = max_caller_silence_seconds
        self.use_speech_detection = use_speech_detection

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        activity = call_activity(test_case, self.use_speech_detection)
        event_key = EVENT_KEYS[activity.source]
        eligible = []
        defects = []
        for index, start in enumerate(agent_starts(activity)):
            if is_opening_line(index, start):
                continue
            key = start.turn if event_key == "turn" else round(start.at, 2)
            if key not in eligible:
                eligible.append(key)
            fired = [
                name
                for name, condition in INTERRUPTION_CONDITIONS.items()
                if condition(start, self)
            ]
            if fired:
                defects.append(
                    VoiceEvent(
                        type=fired[0],
                        **{event_key: key},
                        agent_seconds=round(start.stops_at - start.at, 2),
                        **interruption_place(
                            activity.caller, start.at, start.stops_at
                        ),
                        overlap_seconds=round(start.overlap_seconds, 2),
                        description=AGENT_DESCRIPTIONS[fired[0]].format(
                            overlap=start.overlap_seconds
                        ),
                    )
                )
        return DefectShare(
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
