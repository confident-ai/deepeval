"""Shared pieces of the single-defect voice metrics. Each one scores the
share of agent replies free of its one defect. Caller audio is never judged;
the timing metrics only read when the caller is speaking."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from statistics import median
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from deepeval.metrics.voice.utils.events import (
    RANGE,
    VoiceEvent,
    events_to_dicts,
)
from deepeval.metrics.voice.utils.analysis import (
    AudioMeasurements,
    analyze_audio,
    decode_audio,
)
from deepeval.test_case import ConversationalTestCase
from deepeval.voice.timeline import AudioTimelineEntry, build_audio_timeline

if TYPE_CHECKING:
    from deepeval.metrics.voice.base_metric import VoiceMetricResult

T = TypeVar("T")

# With fewer replies there is no telling which one is off.
MIN_COMPARED_REPLIES = 3
# Brief overlaps between speakers are normal; this also covers the quiet tail
# of the caller's clip.
AGENT_OVERLAP_ALLOWANCE_SECONDS = 0.5
NO_CALL_AUDIO_REASON = "The call has no audio to evaluate."
TURN_AUDIO_NEEDED_REASON = (
    "This metric matches each transcript line to the audio of its own turn, "
    "so it needs audio on each turn; a call recording alone is not enough."
)
STEREO_RECORDING_NEEDED_REASON = (
    "The call recording is not a stereo WAV with the caller and the agent on "
    "separate channels, and no turn has audio of its own."
)
CALL_RECORDING_UNREADABLE_REASON = (
    "The call recording could not be read, and no turn has audio of its own."
)
INAUDIBLE_ERROR = "Reply has no audible sound to measure."
# Where the timing metrics read who spoke when.
CALL_RECORDING = "call_recording"
TURN_TIMESTAMPS = "turn_timestamps"
AGENT_REPLIES = ("agent reply", "agent replies")
CALLER_LINES = ("caller line", "caller lines")
# Where an abrupt cut-off event says the reply stopped, on the call's clock.
STOPPED_AT_SECONDS = "stopped_at_seconds"

# A roll-up metric runs several parts on one call; they share each reply's
# analysis instead of decoding every clip once per part.
_shared_analysis: ContextVar[Optional[Dict[Tuple[str, int], Any]]] = ContextVar(
    "voice_shared_analysis", default=None
)


@contextmanager
def shared_analysis() -> Generator[None, None, None]:
    token = _shared_analysis.set({})
    try:
        yield
    finally:
        _shared_analysis.reset(token)


def _analyzed_once(
    name: str, test_case: ConversationalTestCase, analyze: Callable[[], T]
) -> T:
    cache = _shared_analysis.get()
    if cache is None:
        return analyze()
    key = (name, id(test_case))
    if key not in cache:
        cache[key] = analyze()
    return cache[key]


def agent_reply_measurements(
    test_case: ConversationalTestCase, use_speech_detection: bool = True
) -> Tuple[Dict[float, AudioMeasurements], List[Dict]]:
    # call_activity imports this module, so a top-level import is circular.
    from deepeval.metrics.voice.utils.call_activity import agent_replies

    def analyze() -> Tuple[Dict[float, AudioMeasurements], List[Dict]]:
        replies = agent_replies(test_case, use_speech_detection)
        measurements: Dict[float, AudioMeasurements] = {}
        skipped: List[Dict] = []
        for reply_key, audio in replies.audio.items():
            try:
                measurements[reply_key] = analyze_audio(audio)
            except (TypeError, ValueError) as error:
                skipped.append({**replies.key(reply_key), "error": str(error)})
        return measurements, skipped

    measurements, skipped = _analyzed_once(
        f"agent_reply_measurements:{use_speech_detection}", test_case, analyze
    )
    return dict(measurements), [dict(entry) for entry in skipped]


SPOKEN_REPLY_MIN_SECONDS = 1.0
UNSPOKEN_ERROR = (
    f"Reply has under {SPOKEN_REPLY_MIN_SECONDS:g} second of speech."
)


def is_inaudible(reply: AudioMeasurements) -> bool:
    return reply.silence_fraction >= 1.0


def audible_replies(
    measurements: Dict[float, AudioMeasurements],
    skipped: List[Dict],
    event_key: str = "turn",
) -> Dict[float, AudioMeasurements]:
    replies = {}
    for turn, reply in measurements.items():
        if not is_inaudible(reply):
            replies[turn] = reply
        else:
            skipped.append({event_key: turn, "error": INAUDIBLE_ERROR})
    return replies


def spoken_replies(
    measurements: Dict[float, AudioMeasurements],
    skipped: List[Dict],
    event_key: str = "turn",
) -> Dict[float, AudioMeasurements]:
    """Replies with enough speech to judge its level. A stamped stretch the
    agent barely speaks in has a level, but not one worth comparing."""
    replies = {}
    for turn, reply in measurements.items():
        if reply.speech_span_seconds >= SPOKEN_REPLY_MIN_SECONDS:
            replies[turn] = reply
        else:
            skipped.append({event_key: turn, "error": UNSPOKEN_ERROR})
    return replies


def agent_glitch_events(
    test_case: ConversationalTestCase, use_speech_detection: bool = True
) -> Tuple[List[float], List[VoiceEvent], List[Dict]]:
    # audio_glitch and call_activity import this package, so top-level
    # imports are circular.
    from deepeval.metrics.voice.audio_defects.audio_glitch.utils import (
        detect_audio_glitches,
        detect_recording_reply_glitches,
    )
    from deepeval.metrics.voice.utils.call_activity import agent_replies

    replies = agent_replies(test_case, use_speech_detection)
    if replies.timing_source == CALL_RECORDING:
        report = _analyzed_once(
            f"detect_recording_reply_glitches:{use_speech_detection}",
            test_case,
            lambda: detect_recording_reply_glitches(replies),
        )
    else:
        report = _analyzed_once(
            "detect_audio_glitches",
            test_case,
            lambda: detect_audio_glitches(test_case),
        )
    key = replies.event_key
    agent_reply_keys = [
        reply_key
        for reply_key in report.turns_with_audio
        if reply_key in replies.audio
    ]
    skipped = [
        {key: event.get(key), "error": event.reason}
        for event in report.events
        if event.type == "audio_undecodable"
        and event.get(key) in agent_reply_keys
    ]
    undecodable = {entry[key] for entry in skipped}
    eligible_reply_keys = [
        reply_key
        for reply_key in agent_reply_keys
        if reply_key not in undecodable
    ]
    events = [
        event
        for event in report.events
        if event.get(key) in eligible_reply_keys
    ]
    return eligible_reply_keys, events, skipped


def untimed_call_reason(test_case: ConversationalTestCase) -> Optional[str]:
    if any(
        turn.audio is not None and turn.audio.start_time is None
        for turn in test_case.turns
    ):
        return (
            "Every audio turn needs a start_time to time the call, so none "
            "was inferred."
        )
    return None


def call_has_audio(test_case: ConversationalTestCase) -> bool:
    return any(turn.audio is not None for turn in test_case.turns)


def no_turn_audio_reason(test_case: ConversationalTestCase) -> str:
    if test_case.audio is None:
        return NO_CALL_AUDIO_REASON
    return TURN_AUDIO_NEEDED_REASON


def call_timeline(
    test_case: ConversationalTestCase,
) -> Tuple[List[AudioTimelineEntry], List[Dict], Optional[str]]:
    """The call's audio turns in time order, the agent replies left out because
    their audio could not be measured, or why the call cannot be timed."""
    untimed_reason = untimed_call_reason(test_case)
    if untimed_reason:
        return [], [], untimed_reason
    timeline = build_audio_timeline(test_case.turns)
    timed = {entry.turn_index for entry in timeline}
    skipped = [
        {
            "turn": index,
            "error": "Reply audio could not be decoded to measure its length.",
        }
        for index, turn in enumerate(test_case.turns)
        if turn.role == "assistant"
        and turn.audio is not None
        and index not in timed
    ]
    return timeline, skipped, None


def shifts_from_typical(levels: Dict[int, float]) -> Dict[int, float]:
    """Each reply's distance from the call's typical (median) reply."""
    if not levels:
        return {}
    typical = median(levels.values())
    return {turn: level - typical for turn, level in levels.items()}


def stamped(defect: VoiceEvent, timing_source: Optional[str]) -> VoiceEvent:
    """The defect with where its time came from, and a range mark when the
    metric placed it without saying what the place covers."""
    placed_on_call = (
        defect.start_time is not None or defect.offset_seconds is not None
    )
    return defect.model_copy(
        update={
            "timing_source": defect.timing_source
            or timing_source
            or TURN_TIMESTAMPS,
            "mark": defect.mark or (RANGE if placed_on_call else None),
        }
    )


TURN_TIMESTAMPS_NOTE = (
    " Judged from each turn's own audio and timestamps, as no stereo call "
    "recording was available."
)


def dismissed_note(dismissed: int) -> str:
    if not dismissed:
        return ""
    events = "event was" if dismissed == 1 else "events were"
    return (
        f" {dismissed} flagged {events} dismissed by the audio decision model."
    )


@dataclass
class DefectShare:
    """What a single-defect metric found on a call: the items it judged, the
    ones with its defect, and the ones it could not judge."""

    eligible_turns: Sequence[float]
    defects: List[VoiceEvent]
    skipped: List[Dict]
    defect_label: str
    no_eligible_reason: str = (
        "No decodable assistant audio was available to evaluate."
    )
    min_eligible_turns: int = 1
    counted: Tuple[str, str] = AGENT_REPLIES
    event_key: str = "turn"
    timing_source: Optional[str] = None
    dismissed: int = 0

    def __post_init__(self) -> None:
        self.defects = [
            stamped(defect, self.timing_source) for defect in self.defects
        ]

    @property
    def scored(self) -> bool:
        return len(self.eligible_turns) >= self.min_eligible_turns

    def defect_key(self, defect: VoiceEvent) -> float:
        return getattr(defect, self.event_key)

    def result(self) -> "VoiceMetricResult":
        defective_turns = {self.defect_key(defect) for defect in self.defects}
        breakdown = {
            "eligible_turns": len(self.eligible_turns),
            "turns_with_events": len(defective_turns),
            "events": events_to_dicts(
                sorted(self.defects, key=self.defect_key)
            ),
            "skipped": self.skipped,
        }
        if self.timing_source is not None:
            breakdown["timing_source"] = self.timing_source
        if self.dismissed:
            breakdown["dismissed_events"] = self.dismissed
        if not self.scored:
            return None, self.no_eligible_reason, breakdown
        if not defective_turns:
            reason = f"No {self.counted[0]} had {self.defect_label}."
        else:
            reason = (
                f"{len(defective_turns)} of {len(self.eligible_turns)} "
                f"{self.counted[1]} had {self.defect_label}."
            )
        if self.timing_source == TURN_TIMESTAMPS:
            reason += TURN_TIMESTAMPS_NOTE
        reason += dismissed_note(self.dismissed)
        score = 1.0 - len(defective_turns) / len(self.eligible_turns)
        return score, reason, breakdown
