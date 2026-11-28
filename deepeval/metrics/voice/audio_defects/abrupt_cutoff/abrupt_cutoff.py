from typing import List, Optional

from deepeval.metrics.voice.audio_defects.dead_stream.dead_stream import (
    dead_streams,
)

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
    call_timeline,
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
# A cut-off ends the reply. The agent going dead and then carrying on is a
# hole in the reply, which is Dead Stream's to judge.
REPLY_END_SECONDS = 0.05
DEAD_STREAM_AFTER_STOP_SECONDS = 0.3
# A reply whose text ends a sentence was said in full, however sharply its
# last sound stops. The stamps around a stop can be this far off.
SENTENCE_ENDS = (".", "?", "!", "…")
CLOSING_MARKS = "\"'”’)]"
STAMP_BEFORE_STOP_SECONDS = 0.5
STAMP_AFTER_STOP_SECONDS = 1.5


def _said_in_full(text: str) -> bool:
    if not any(character.isalnum() for character in text):
        return False
    return text.strip().rstrip(CLOSING_MARKS).endswith(SENTENCE_ENDS)


def _reply_text_at(
    test_case: ConversationalTestCase, stopped_at: float
) -> Optional[str]:
    """The text of the agent reply whose audio the stop falls in."""
    timeline, _, _ = call_timeline(test_case)
    replies = [
        entry
        for entry in timeline
        if entry.role == "assistant"
        and entry.start_time - STAMP_BEFORE_STOP_SECONDS
        <= stopped_at
        <= entry.end_time + STAMP_AFTER_STOP_SECONDS
    ]
    if not replies:
        return None
    return test_case.turns[replies[-1].turn_index].content


DEFECT_LABEL = "an abrupt cut-off"


def _resumes_after(holes: List[VoiceEvent], stopped_at: float) -> bool:
    return any(
        hole.start_time is not None
        and abs(hole.start_time - stopped_at) <= DEAD_STREAM_AFTER_STOP_SECONDS
        for hole in holes
    )


def _recording_share(
    recording: CallActivity,
    holes: List[VoiceEvent],
    test_case: ConversationalTestCase,
) -> DefectShare:
    replies = stretches(recording.agent, AGENT_PAUSE_SECONDS)
    defects = []
    for reply in replies:
        cut = next(
            (
                stop
                for stop in recording.agent_stops
                if reply.end - REPLY_END_SECONDS <= stop.at <= reply.end
                and stop.ends_abruptly
                and not _resumes_after(holes, stop.at)
                and not _said_in_full(_reply_text_at(test_case, stop.at) or "")
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
        holes = dead_streams(test_case, self.use_speech_detection)
        if recording is not None:
            return _recording_share(recording, holes, test_case)
        measurements, skipped = agent_reply_measurements(test_case)
        _, events, _ = agent_glitch_events(test_case)
        stream_cuts = {
            event.turn: event.model_copy(
                update={"mark": MOMENT, "description": CUT_DESCRIPTION}
            )
            for event in events
            if event.type == "stream_cut"
        }
        cuts = [
            stream_cuts.get(turn, _cut_event(test_case, turn, reply))
            for turn, reply in measurements.items()
            if (turn in stream_cuts or reply.ends_abruptly)
            and not test_case.turns[turn].interrupted
            and not _said_in_full(test_case.turns[turn].content)
        ]
        defects = [
            cut
            for cut in cuts
            if cut.start_time is None
            or not _resumes_after(holes, cut.start_time + CUT_EDGE_SECONDS)
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
