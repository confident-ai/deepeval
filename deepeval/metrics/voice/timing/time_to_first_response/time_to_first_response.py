from typing import Optional

from deepeval.metrics.voice.utils.call_activity import (
    SPEAKER_PAUSE_SECONDS,
    CallActivity,
    call_recording_activity,
    stretches,
)
from deepeval.metrics.voice.utils.defects import (
    CALL_RECORDING,
    TURN_TIMESTAMPS,
    defect_share_result,
    untimed_call_reason,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase


class TimeToFirstResponseMetric(BaseVoiceMetric):
    # The 2 s latency limit plus 1 s for the line to connect, since the clock
    # starts when the call connects.
    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        max_wait_seconds: int = 3,
        use_speech_detection: bool = True,
        flaky: bool = False,
    ):
        if max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be greater than 0.")
        super().__init__(
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            flaky=flaky,
        )
        self.max_wait_seconds = max_wait_seconds
        self.use_speech_detection = use_speech_detection

    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        recording = call_recording_activity(
            test_case, self.use_speech_detection
        )
        if recording is not None:
            return self._evaluate_recording(recording)
        defect_label = (
            f"a first response over {self.max_wait_seconds} seconds after "
            "the call connected"
        )
        untimed_reason = untimed_call_reason(test_case)
        audio_turns = (
            []
            if untimed_reason
            else sorted(
                (turn.audio.start_time, index)
                for index, turn in enumerate(test_case.turns)
                if turn.audio is not None
            )
        )
        if not audio_turns:
            return defect_share_result(
                [],
                [],
                [],
                defect_label,
                no_eligible_reason=untimed_reason
                or "The call had no timed audio.",
                timing_source=TURN_TIMESTAMPS,
            )
        greeting_at, greeting_turn = audio_turns[0]
        if test_case.turns[greeting_turn].role != "assistant":
            return defect_share_result(
                [],
                [],
                [],
                defect_label,
                no_eligible_reason="The caller spoke first, so the agent was "
                "not expected to greet.",
                timing_source=TURN_TIMESTAMPS,
            )
        defects = []
        if greeting_at > self.max_wait_seconds:
            defects.append(
                {
                    "type": "slow_first_response",
                    "turn": greeting_turn,
                    "first_response_seconds": round(greeting_at, 2),
                }
            )
        return defect_share_result(
            [greeting_turn],
            defects,
            [],
            defect_label,
            timing_source=TURN_TIMESTAMPS,
        )

    def _evaluate_recording(self, recording: CallActivity) -> VoiceMetricResult:
        if not recording.agent:
            return defect_share_result(
                [],
                [],
                [],
                "a slow first response",
                no_eligible_reason="The agent made no sound on the call "
                "recording.",
                event_key="at_seconds",
                timing_source=CALL_RECORDING,
            )
        agent_at = recording.agent[0].start
        caller_first = next(
            iter(stretches(recording.caller, SPEAKER_PAUSE_SECONDS)), None
        )
        if caller_first is not None and caller_first.start < agent_at:
            wait = max(0.0, agent_at - caller_first.end)
            defect_label = (
                f"a first response over {self.max_wait_seconds} seconds after "
                "the caller's first words"
            )
        else:
            wait = agent_at
            defect_label = (
                f"a first response over {self.max_wait_seconds} seconds after "
                "the call connected"
            )
        at = round(agent_at, 2)
        defects = []
        if wait > self.max_wait_seconds:
            defects.append(
                {
                    "type": "slow_first_response",
                    "at_seconds": at,
                    "first_response_seconds": round(wait, 2),
                }
            )
        return defect_share_result(
            [at],
            defects,
            [],
            defect_label,
            event_key="at_seconds",
            timing_source=CALL_RECORDING,
        )

    @property
    def __name__(self):
        return "Time to First Response"
