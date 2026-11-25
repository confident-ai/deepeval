from typing import Optional

from deepeval.metrics.voice.audio_defects.echo.utils import (
    ECHO_MIN_REPLY_SECONDS,
    speaker_leak,
)
from deepeval.metrics.voice.utils.call_activity import (
    read_call_recording,
    recording_activity,
)
from deepeval.metrics.voice.utils.defects import (
    CALL_RECORDING,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.config.voice_eval_mode import VoiceEvalModeName
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import (
    WHOLE_REPLY,
    VoiceEvent,
    on_call,
)

DEFECT_LABEL = "an echo"


class EchoMetric(BaseDefectMetric):
    event_question = "Can the AI agent's own voice be heard echoed back on this caller's line?"
    event_speaker = "user"

    def __init__(
        self,
        *,
        threshold: Optional[float] = 0.5,
        include_reason: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
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

    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        agent_channel = test_case.agent_channel
        channels = read_call_recording(test_case.audio, agent_channel)
        # Agent sound is read the same way with or without speech detection.
        activity = recording_activity(test_case.audio, False, agent_channel)
        if channels is None or activity is None:
            return DefectShare(
                [],
                [],
                [],
                DEFECT_LABEL,
                no_eligible_reason="Echo needs a stereo call recording "
                "(the test case's audio) with the caller on one channel and "
                "the agent on the other.",
                event_key="reply_start",
            )
        caller, agent, sample_rate = channels
        eligible = []
        defects = []
        for span in activity.agent:
            if span.end - span.start < ECHO_MIN_REPLY_SECONDS:
                continue
            at = round(span.start, 2)
            eligible.append(at)
            leak = speaker_leak(
                agent, caller, sample_rate, span.start, span.end
            )
            if leak is not None:
                delay, level = leak
                defects.append(
                    VoiceEvent(
                        type="echo",
                        reply_start=at,
                        delay_seconds=round(delay, 3),
                        level_db=round(level, 1),
                        mark=WHOLE_REPLY,
                        description="Agent's voice echoed back from the "
                        "caller's side",
                        **on_call(span.start, span.end - span.start),
                    )
                )
        return DefectShare(
            eligible,
            defects,
            [],
            DEFECT_LABEL,
            no_eligible_reason="The agent never spoke for "
            f"{ECHO_MIN_REPLY_SECONDS:g} seconds or more on the call recording.",
            event_key="reply_start",
            timing_source=CALL_RECORDING,
        )

    @property
    def __name__(self):
        return "Echo"
