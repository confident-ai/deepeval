from typing import Optional

from deepeval.metrics.voice.utils.call_activity import (
    SPEAKER_PAUSE_SECONDS,
    CallActivity,
    call_recording_activity,
    stretches,
    turn_activity,
)
from deepeval.metrics.voice.utils.defects import (
    CALL_RECORDING,
    TURN_TIMESTAMPS,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent, on_call


class TimeToFirstResponseMetric(BaseDefectMetric):
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

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        recording = call_recording_activity(
            test_case, self.use_speech_detection
        )
        if recording is not None:
            return self._recording_share(recording)
        defect_label = (
            f"a first response over {self.max_wait_seconds} seconds after "
            "the call connected"
        )
        activity = turn_activity(test_case, self.use_speech_detection)
        heard = [(span.start, span.turn) for span in activity.agent]
        heard += [(span.start, span.turn) for span in activity.caller]
        timed = {turn for _, turn in heard}
        # Audio that can't be decoded still has a start to time it by.
        heard += [
            (turn.audio.start_time, index)
            for index, turn in enumerate(test_case.turns)
            if index not in timed
            and turn.audio is not None
            and turn.audio.start_time is not None
            and activity.untimed_reason is None
        ]
        if not heard:
            return DefectShare(
                [],
                [],
                [],
                defect_label,
                no_eligible_reason=activity.untimed_reason
                or "The call had no timed audio.",
                timing_source=TURN_TIMESTAMPS,
            )
        greeting_at, greeting_turn = min(heard)
        if test_case.turns[greeting_turn].role != "assistant":
            return DefectShare(
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
                VoiceEvent(
                    type="slow_first_response",
                    turn=greeting_turn,
                    first_response_seconds=round(greeting_at, 2),
                    description=f"Agent took {greeting_at:.1f}s to first "
                    "respond",
                    **on_call(0.0, greeting_at),
                )
            )
        return DefectShare(
            [greeting_turn],
            defects,
            [],
            defect_label,
            timing_source=TURN_TIMESTAMPS,
        )

    def _recording_share(self, recording: CallActivity) -> DefectShare:
        if not recording.agent:
            return DefectShare(
                [],
                [],
                [],
                "a slow first response",
                no_eligible_reason="The agent made no sound on the call "
                "recording.",
                event_key="reply_start",
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
                VoiceEvent(
                    type="slow_first_response",
                    reply_start=at,
                    first_response_seconds=round(wait, 2),
                    description=f"Agent took {wait:.1f}s to first respond",
                    **on_call(agent_at - wait, wait),
                )
            )
        return DefectShare(
            [at],
            defects,
            [],
            defect_label,
            event_key="reply_start",
            timing_source=CALL_RECORDING,
        )

    @property
    def __name__(self):
        return "Time to First Response"
