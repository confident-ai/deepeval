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
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.events import VoiceEvent, on_call

DEFECT_LABEL = "an echo"


class EchoMetric(BaseDefectMetric):
    def defect_share(self, test_case: ConversationalTestCase) -> DefectShare:
        path = test_case.call_recording_path
        channels = read_call_recording(path)
        # Agent sound is read the same way with or without speech detection.
        activity = recording_activity(path, False)
        if channels is None or activity is None:
            return DefectShare(
                [],
                [],
                [],
                DEFECT_LABEL,
                no_eligible_reason="Echo needs a stereo call recording "
                "(call_recording_path) with the caller on the left channel "
                "and the agent on the right.",
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
                        **on_call(span.start + delay, span.end - span.start),
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
