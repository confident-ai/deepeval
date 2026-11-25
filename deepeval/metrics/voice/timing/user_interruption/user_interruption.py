from typing import Callable, Dict, Optional

from deepeval.metrics.voice.utils.call_activity import (
    BACKCHANNEL_MAX_SECONDS,
    CALLER_EVENT_KEYS,
    COLLISION_SECONDS,
    FALSE_START_MAX_SECONDS,
    TALK_OVER_MIN_SECONDS,
    CallerStart,
    call_activity,
    caller_starts,
    interruption_place,
)
from deepeval.metrics.voice.utils.defects import DefectShare
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.config.voice_eval_mode import VoiceEvalModeName
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent


def agent_had_the_floor(start: CallerStart) -> bool:
    """The agent is talking AND started more than 0.5 s before the caller;
    closer starts are a collision, which Agent Interruption judges."""
    return (
        start.agent_talking
        and start.agent_talking_for is not None
        and start.agent_talking_for > COLLISION_SECONDS
    )


def is_false_start(start: CallerStart) -> bool:
    """The agent had the floor AND the caller stops within 1.5 s."""
    return (
        agent_had_the_floor(start)
        and start.stops_at - start.at <= FALSE_START_MAX_SECONDS
    )


def is_talk_over(start: CallerStart) -> bool:
    """The agent had the floor AND both keep talking for 1 s or more."""
    return (
        agent_had_the_floor(start)
        and start.overlap_seconds >= TALK_OVER_MIN_SECONDS
    )


# Interruption == False start OR Talk-over. Add a condition by adding it here.
INTERRUPTION_CONDITIONS: Dict[str, Callable[[CallerStart], bool]] = {
    "false_start": is_false_start,
    "talk_over": is_talk_over,
}

CALLER_DESCRIPTIONS = {
    "false_start": "Caller started over the agent, then stopped",
    "talk_over": "Caller talked over the agent for {overlap:.1f}s",
}


class UserInterruptionMetric(BaseDefectMetric):
    event_question = "Is a person speaking in this clip, rather than only background noise, breathing or silence?"
    event_speaker = "user"

    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        use_speech_detection: bool = True,
        audio_decision_model: Optional[DeepEvalBaseAudioDecisionModel] = None,
        eval_mode: Optional[VoiceEvalModeName] = None,
        flaky: bool = False,
    ):
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self._use_audio_decision_model(audio_decision_model, eval_mode)
        self.use_speech_detection = use_speech_detection

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        activity = call_activity(test_case, self.use_speech_detection)
        event_key = CALLER_EVENT_KEYS[activity.source]
        eligible = []
        defects = []
        for start in caller_starts(activity) if activity.agent else []:
            if start.stops_at - start.at < BACKCHANNEL_MAX_SECONDS:
                continue
            key = start.turn if event_key == "turn" else round(start.at, 2)
            if key not in eligible:
                eligible.append(key)
            fired = [
                name
                for name, condition in INTERRUPTION_CONDITIONS.items()
                if condition(start)
            ]
            if fired:
                defects.append(
                    VoiceEvent(
                        type=fired[0],
                        **{event_key: key},
                        caller_seconds=round(start.stops_at - start.at, 2),
                        **interruption_place(
                            activity.agent, start.at, start.stops_at
                        ),
                        overlap_seconds=round(start.overlap_seconds, 2),
                        description=CALLER_DESCRIPTIONS[fired[0]].format(
                            overlap=start.overlap_seconds
                        ),
                    )
                )
        return DefectShare(
            eligible,
            defects,
            list(activity.skipped),
            "the caller interrupting the agent",
            no_eligible_reason=activity.untimed_reason
            or (
                "The caller never spoke for "
                f"{BACKCHANNEL_MAX_SECONDS:g} second or more."
                if activity.agent
                else "The agent made no sound, so there was nothing to interrupt."
            ),
            counted=("caller turn", "caller turns"),
            event_key=event_key,
            timing_source=activity.source,
        )

    @property
    def __name__(self):
        return "User Interruption"
