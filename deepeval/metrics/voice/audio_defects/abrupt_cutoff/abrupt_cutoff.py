from deepeval.metrics.voice.utils.call_activity import (
    AGENT_PAUSE_SECONDS,
    CallActivity,
    call_recording_activity,
    sound_between,
    stretches,
)
from deepeval.metrics.voice.utils.defects import (
    CALL_RECORDING,
    TURN_TIMESTAMPS,
    agent_glitch_events,
    agent_reply_measurements,
    defect_share_result,
)
from deepeval.metrics.voice.base_metric import (
    BaseVoiceMetric,
    VoiceMetricResult,
)
from deepeval.test_case import ConversationalTestCase

# The agent stopping because the caller spoke up is not a cut-off.
CALLER_BEFORE_STOP_SECONDS = 1.0
CALLER_AFTER_STOP_SECONDS = 0.2
CALLER_SILENT_MAX_SECONDS = 0.2
# Sound still going when the recording ends is Unexpected Call End's to judge.
RECORDING_END_SECONDS = 0.05
DEFECT_LABEL = "an abrupt cut-off"


def _recording_result(recording: CallActivity) -> VoiceMetricResult:
    replies = stretches(recording.agent, AGENT_PAUSE_SECONDS)
    defects = []
    for reply in replies:
        cut = next(
            (
                stop
                for stop in recording.agent_stops
                if reply.start < stop.at <= reply.end
                and stop.ends_abruptly
                and stop.at < recording.duration - RECORDING_END_SECONDS
                and sound_between(
                    recording.caller,
                    stop.at - CALLER_BEFORE_STOP_SECONDS,
                    stop.at + CALLER_AFTER_STOP_SECONDS,
                )
                < CALLER_SILENT_MAX_SECONDS
            ),
            None,
        )
        if cut is not None:
            defects.append(
                {
                    "type": "abrupt_cutoff",
                    "at_seconds": round(reply.start, 2),
                    "stopped_at_seconds": round(cut.at, 2),
                }
            )
    return defect_share_result(
        [round(reply.start, 2) for reply in replies],
        defects,
        [],
        DEFECT_LABEL,
        no_eligible_reason="The agent made no sound on the call recording.",
        event_key="at_seconds",
        timing_source=CALL_RECORDING,
    )


class AbruptCutoffMetric(BaseVoiceMetric):
    def _evaluate(self, test_case: ConversationalTestCase) -> VoiceMetricResult:
        recording = call_recording_activity(test_case)
        if recording is not None:
            return _recording_result(recording)
        measurements, skipped = agent_reply_measurements(test_case)
        _, events, _ = agent_glitch_events(test_case)
        stream_cuts = {
            event["turn"]: event
            for event in events
            if event["type"] == "stream_cut"
        }
        defects = [
            stream_cuts.get(turn, {"type": "abrupt_cutoff", "turn": turn})
            for turn, reply in measurements.items()
            if (turn in stream_cuts or reply.ends_abruptly)
            and not test_case.turns[turn].interrupted
        ]
        return defect_share_result(
            list(measurements),
            defects,
            skipped,
            DEFECT_LABEL,
            timing_source=TURN_TIMESTAMPS,
        )

    @property
    def __name__(self):
        return "Abrupt Cut-off"
