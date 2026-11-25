from typing import Optional

from deepeval.metrics.voice.utils.call_activity import (
    AGENT_PAUSE_SECONDS,
    CallActivity,
    call_recording_activity,
    sound_between,
    stretches,
)
from deepeval.metrics.voice.utils.defects import (
    CALL_RECORDING,
    STOPPED_AT_SECONDS,
    TURN_TIMESTAMPS,
    agent_glitch_events,
    agent_reply_measurements,
    DefectShare,
)
from deepeval.metrics.voice.base_metric import BaseDefectMetric
from deepeval.config.voice_eval_mode import VoiceEvalModeName
from deepeval.models.base_model import DeepEvalBaseAudioDecisionModel
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics.voice.utils.analysis import EDGE_MS
from deepeval.metrics.voice.utils.events import (
    MOMENT,
    VoiceEvent,
    on_call,
    placed,
)

# The cut is judged on the last 20 ms before the sound stops.
CUT_EDGE_SECONDS = EDGE_MS / 1000
CUT_DESCRIPTION = "Agent's audio cut off abruptly"


def _cut_event(test_case, turn, reply) -> VoiceEvent:
    if reply.sound_end_seconds is None:
        return VoiceEvent(
            type="abrupt_cutoff",
            turn=turn,
            mark=MOMENT,
            description=CUT_DESCRIPTION,
        )
    return VoiceEvent(
        type="abrupt_cutoff",
        turn=turn,
        mark=MOMENT,
        description=CUT_DESCRIPTION,
        **placed(
            test_case.turns[turn].audio,
            max(0.0, reply.sound_end_seconds - CUT_EDGE_SECONDS),
            min(CUT_EDGE_SECONDS, reply.sound_end_seconds),
        ),
    )


# The agent stopping because the caller spoke up is not a cut-off.
CALLER_BEFORE_STOP_SECONDS = 1.0
CALLER_AFTER_STOP_SECONDS = 0.2
CALLER_SILENT_MAX_SECONDS = 0.2
# Sound still going when the recording ends is Unexpected Call End's to judge.
RECORDING_END_SECONDS = 0.05
DEFECT_LABEL = "an abrupt cut-off"


def _recording_share(recording: CallActivity) -> DefectShare:
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
                VoiceEvent(
                    type="abrupt_cutoff",
                    reply_start=round(reply.start, 2),
                    mark=MOMENT,
                    description=CUT_DESCRIPTION,
                    **{STOPPED_AT_SECONDS: round(cut.at, 2)},
                    **on_call(cut.at - CUT_EDGE_SECONDS, CUT_EDGE_SECONDS),
                )
            )
    return DefectShare(
        [round(reply.start, 2) for reply in replies],
        defects,
        [],
        DEFECT_LABEL,
        no_eligible_reason="The agent made no sound on the call recording.",
        event_key="reply_start",
        timing_source=CALL_RECORDING,
    )


class AbruptCutoffMetric(BaseDefectMetric):
    event_question = (
        "Does the speaker's voice cut off abruptly, mid-word or mid-sentence?"
    )
    event_speaker = "assistant"

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
        recording = call_recording_activity(
            test_case, self.use_speech_detection
        )
        if recording is not None:
            return _recording_share(recording)
        measurements, skipped = agent_reply_measurements(test_case)
        _, events, _ = agent_glitch_events(test_case)
        stream_cuts = {
            event.turn: event.model_copy(
                update={"mark": MOMENT, "description": CUT_DESCRIPTION}
            )
            for event in events
            if event.type == "stream_cut"
        }
        defects = [
            stream_cuts.get(turn, _cut_event(test_case, turn, reply))
            for turn, reply in measurements.items()
            if (turn in stream_cuts or reply.ends_abruptly)
            and not test_case.turns[turn].interrupted
        ]
        return DefectShare(
            list(measurements),
            defects,
            skipped,
            DEFECT_LABEL,
            timing_source=TURN_TIMESTAMPS,
        )

    @property
    def __name__(self):
        return "Abrupt Cut-off"
