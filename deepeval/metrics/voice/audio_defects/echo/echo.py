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
    defect_share_result,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase

DEFECT_LABEL = "an echo"


class EchoMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        path = test_case.call_recording_path
        channels = read_call_recording(path)
        # Agent sound is read the same way with or without speech detection.
        activity = recording_activity(path, False)
        if channels is None or activity is None:
            return defect_share_result(
                [],
                [],
                [],
                DEFECT_LABEL,
                no_eligible_reason="Echo needs a stereo call recording "
                "(call_recording_path) with the caller on the left channel "
                "and the agent on the right.",
                event_key="at_seconds",
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
                    {
                        "type": "echo",
                        "at_seconds": at,
                        "delay_seconds": round(delay, 3),
                        "level_db": round(level, 1),
                    }
                )
        return defect_share_result(
            eligible,
            defects,
            [],
            DEFECT_LABEL,
            no_eligible_reason="The agent never spoke for "
            f"{ECHO_MIN_REPLY_SECONDS:g} seconds or more on the call recording.",
            event_key="at_seconds",
            timing_source=CALL_RECORDING,
        )

    @property
    def __name__(self):
        return "Echo"
