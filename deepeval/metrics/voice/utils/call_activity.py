"""When each speaker made sound on a call. With a stereo recording of the
whole call (caller left, agent right) it comes from each channel's loudness,
and with speech detection on, the caller must also be heard as speaking.
Without one, each turn's own audio is placed by its timestamp and only the
stretches inside it where the speaker is heard count, judged the same way for
both speakers."""

import hashlib
import threading
import wave
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from deepeval.metrics.voice.utils.analysis import (
    CONTIGUOUS_CLIP_TOLERANCE_SECONDS,
    STOP_CONTEXT_SECONDS,
    speech_and_noise_levels,
    decode_audio,
    ends_abruptly,
    frame_samples,
    get_numpy,
    rms,
)
from deepeval.metrics.voice.utils.defects import (
    AGENT_OVERLAP_ALLOWANCE_SECONDS,
    CALL_RECORDING,
    TURN_TIMESTAMPS,
    call_timeline,
)
from deepeval.metrics.voice.utils.events import on_call
from deepeval.metrics.voice.utils.speech_detection import speech_frames
from deepeval.test_case import Audio, ConversationalTestCase, Turn
from deepeval.voice.connectors import audio_utils
from deepeval.voice.timeline import AudioTimelineEntry, build_audio_timeline

T = TypeVar("T")

EVENT_KEYS = {CALL_RECORDING: "reply_start", TURN_TIMESTAMPS: "turn"}
CALLER_EVENT_KEYS = {CALL_RECORDING: "caller_start", TURN_TIMESTAMPS: "turn"}
# Gaps shorter than this sit between syllables, not between utterances.
SOUND_GAP_SECONDS = 0.3
# Shorter bursts are clicks and breaths.
MIN_SOUND_SECONDS = 0.15
# A pause shorter than this doesn't end a stretch of the agent's speech.
AGENT_PAUSE_SECONDS = 1.0
# The same for the caller, so caller starts mirror agent starts.
CALLER_PAUSE_SECONDS = 1.0
# Speakers starting within this of each other collided; Agent Interruption
# owns the moment, so it is never also a User Interruption.
COLLISION_SECONDS = 0.5
# An interruption either backs off within this or keeps talking over for this.
FALSE_START_MAX_SECONDS = 1.5
TALK_OVER_MIN_SECONDS = 1.0
# Shorter caller sounds are an "mm-hm" while listening, not a turn.
BACKCHANNEL_MAX_SECONDS = 1.0
# A speaker takes the floor after at least this much of their own silence.
SPEAKER_PAUSE_SECONDS = 0.5
# The caller is already talking when this much of the moment before the agent
# starts holds caller sound.
CALLER_TALKING_WINDOW_SECONDS = 0.3
CALLER_TALKING_MIN_SECONDS = 0.2
# Room past the last 20 ms frame to find where the sound really stops.
STOP_MARGIN_SECONDS = 0.02
# Speech this close to the edge of its turn's audio runs to the edge, so the
# silence between two touching turns can't be heard.
SPEECH_AT_EDGE_SECONDS = 0.05
# A turn's own audio can hold a short, quiet word, like a faint "yes" or one
# cut in half where the clips meet. What the speech model hears counts down
# to this level and length there; loudness alone keeps the defaults above.
TURN_SPEECH_MIN_LEVEL = 100.0
TURN_SPEECH_MIN_SECONDS = 0.06
RECORDING_CACHE_SIZE = 32
TURN_AUDIO_CACHE_SIZE = 512


@dataclass(frozen=True)
class Span:
    start: float
    end: float
    turn: Optional[int] = None


@dataclass(frozen=True)
class SoundStop:
    at: float
    ends_abruptly: bool


@dataclass(frozen=True)
class CallActivity:
    source: str
    agent: Tuple[Span, ...]
    caller: Tuple[Span, ...]
    duration: float
    agent_stops: Tuple[SoundStop, ...] = ()
    skipped: Tuple[Dict, ...] = ()
    untimed_reason: Optional[str] = None


@dataclass(frozen=True)
class AgentStart:
    """The agent starting to speak after at least AGENT_PAUSE_SECONDS of its
    own silence, and what the caller was doing around it."""

    at: float
    stops_at: float
    turn: Optional[int]
    caller_talking: bool
    caller_starts_after: Optional[float]
    caller_silent_for: Optional[float]
    caller_pause_seconds: Optional[float]
    caller_runs_past: float
    caller_spoke_last: bool
    overlap_seconds: float


@dataclass(frozen=True)
class CallerStart:
    """The caller starting to speak after at least CALLER_PAUSE_SECONDS of
    their own silence, and what the agent was doing around it."""

    at: float
    stops_at: float
    turn: Optional[int]
    agent_talking: bool
    agent_talking_for: Optional[float]
    overlap_seconds: float


def stretches(spans: Sequence[Span], own_silence: float) -> List[Span]:
    merged: List[Span] = []
    for span in spans:
        if merged and span.start - merged[-1].end < own_silence:
            merged[-1] = Span(
                merged[-1].start, max(merged[-1].end, span.end), merged[-1].turn
            )
        else:
            merged.append(span)
    return merged


def _continues(
    speech: Dict[int, Tuple[float, float]], previous: int, index: int
) -> bool:
    before, after = speech.get(previous), speech.get(index)
    return (
        before is not None
        and after is not None
        and after[0] - before[1] < CALLER_PAUSE_SECONDS
    )


def caller_lines(
    turns: Sequence[Turn], use_speech_detection: bool
) -> List[Tuple[int, int]]:
    """First and last turn of each caller line. Caller turns whose speech
    starts within CALLER_PAUSE_SECONDS of the previous one's speech ending
    are one line, split by the transcriber, not the caller speaking again."""
    speech = turn_speech(turns, use_speech_detection)
    lines: List[Tuple[int, int]] = []
    for index, turn in enumerate(turns):
        if turn.role != "user":
            continue
        if (
            lines
            and lines[-1][1] == index - 1
            and _continues(speech, index - 1, index)
        ):
            lines[-1] = (lines[-1][0], index)
        else:
            lines.append((index, index))
    return lines


def silence_after_line(
    turns: Sequence[Turn], last: int, use_speech_detection: bool
) -> Dict[str, Any]:
    """The silence after a caller line on the call's clock: from where its
    last turn's speech ends to where the next turn's speech starts, or to the
    end of the call's audio when no turn follows."""
    timeline = build_audio_timeline(turns)
    speech = _speech_by_turn(_heard(timeline, use_speech_detection))
    line = speech.get(last)
    if line is None:
        return {}
    line_end = line[1]
    following = [
        start
        for turn, (start, _) in speech.items()
        if turn > last and start >= line_end
    ]
    end = min(following, default=max(entry.end_time for entry in timeline))
    return on_call(line_end, max(0.0, end - line_end))


def call_end(turns: Sequence[Turn]) -> Optional[float]:
    """Where the call's last timed audio ends."""
    timeline = build_audio_timeline(turns)
    return max((entry.end_time for entry in timeline), default=None)


def interruption_place(
    other: Sequence[Span], start: float, end: float
) -> Dict[str, Any]:
    """Where both speakers talked during an interruption, on the call's clock;
    the interrupter's own stretch when the other never spoke inside it."""
    overlaps = [
        (max(span.start, start), min(span.end, end))
        for span in other
        if span.start < end and span.end > start
    ]
    if not overlaps:
        return on_call(start, end - start)
    first = min(overlap[0] for overlap in overlaps)
    last = max(overlap[1] for overlap in overlaps)
    return on_call(first, last - first)


def sound_between(spans: Sequence[Span], start: float, end: float) -> float:
    return sum(
        max(0.0, min(span.end, end) - max(span.start, start)) for span in spans
    )


def last_sound_end(spans: Sequence[Span], before: float) -> Optional[float]:
    """Where the latest sound that started before `before` ends; past
    `before` when that sound is still going."""
    return max(
        (span.end for span in spans if span.start < before), default=None
    )


def _pause_around(
    speaker_stretches: Sequence[Span], at: float
) -> Optional[float]:
    """A speaker's pause around `at`, from when they stopped speaking to when
    they started again; 0 when they had already been talking."""
    for index, stretch in enumerate(speaker_stretches):
        if stretch.end <= at:
            continue
        if stretch.start < at - CALLER_TALKING_WINDOW_SECONDS:
            return 0.0
        if index == 0:
            return None
        return stretch.start - speaker_stretches[index - 1].end
    return None


def _is_talking(spans: Sequence[Span], at: float) -> bool:
    return (
        sound_between(spans, at - CALLER_TALKING_WINDOW_SECONDS, at)
        >= CALLER_TALKING_MIN_SECONDS
    )


def _next_start(stretches_: Sequence[Span], at: float) -> Optional[float]:
    """Seconds from `at` until the next stretch starts; 0 when one is
    already going."""
    return next(
        (max(at, span.start) - at for span in stretches_ if span.end > at),
        None,
    )


def is_opening_line(index: int, start: AgentStart) -> bool:
    return index == 0 and start.caller_silent_for is None


def agent_starts(activity: CallActivity) -> List[AgentStart]:
    caller = activity.caller
    caller_stretches = stretches(caller, SPEAKER_PAUSE_SECONDS)
    starts: List[AgentStart] = []
    previous_end: Optional[float] = None
    for stretch in stretches(activity.agent, AGENT_PAUSE_SECONDS):
        at = stretch.start
        caller_end = last_sound_end(caller, at)
        starts.append(
            AgentStart(
                at=at,
                stops_at=stretch.end,
                turn=stretch.turn,
                caller_talking=_is_talking(caller, at),
                caller_starts_after=_next_start(caller_stretches, at),
                caller_silent_for=(
                    None if caller_end is None else max(0.0, at - caller_end)
                ),
                caller_pause_seconds=_pause_around(caller_stretches, at),
                caller_runs_past=(
                    0.0 if caller_end is None else max(0.0, caller_end - at)
                ),
                caller_spoke_last=caller_end is not None
                and (previous_end is None or caller_end > previous_end),
                overlap_seconds=sound_between(caller, at, stretch.end),
            )
        )
        previous_end = stretch.end
    return starts


def caller_starts(activity: CallActivity) -> List[CallerStart]:
    agent = activity.agent
    # The same stretches whose starts Agent Interruption judges.
    agent_stretches = stretches(agent, AGENT_PAUSE_SECONDS)
    starts: List[CallerStart] = []
    for stretch in stretches(activity.caller, CALLER_PAUSE_SECONDS):
        at = stretch.start
        agent_since = next(
            (
                span.start
                for span in agent_stretches
                if span.start < at < span.end
            ),
            None,
        )
        starts.append(
            CallerStart(
                at=at,
                stops_at=stretch.end,
                turn=stretch.turn,
                agent_talking=_is_talking(agent, at),
                agent_talking_for=(
                    None if agent_since is None else at - agent_since
                ),
                overlap_seconds=sound_between(agent, at, stretch.end),
            )
        )
    return starts


def _loud_frames(levels: Sequence[float]) -> List[bool]:
    return [level >= audio_utils.DEFAULT_SILENCE_RMS for level in levels]


def _channel_spans(
    talking: Sequence[bool], min_seconds: float = MIN_SOUND_SECONDS
) -> List[Span]:
    frame_seconds = audio_utils.DEFAULT_FRAME_MS / 1000
    bounds: List[List[float]] = []
    for index, is_talking in enumerate(talking):
        if not is_talking:
            continue
        start = index * frame_seconds
        if bounds and start - bounds[-1][1] < SOUND_GAP_SECONDS:
            bounds[-1][1] = start + frame_seconds
        else:
            bounds.append([start, start + frame_seconds])
    return [
        Span(start, end) for start, end in bounds if end - start >= min_seconds
    ]


def _frame_levels(samples: Any, sample_rate: int) -> List[float]:
    return [
        rms(frame)
        for frame in frame_samples(
            samples, sample_rate, audio_utils.DEFAULT_FRAME_MS
        )
    ]


def _read_channels(
    wav: bytes, agent_channel: str = "right"
) -> Optional[Tuple[Any, Any, int]]:
    try:
        pcm, sample_rate, channels = audio_utils.wav_bytes_to_pcm16(wav)
    except (EOFError, ValueError, wave.Error):
        return None
    if channels != 2 or not sample_rate:
        return None
    np = get_numpy()
    frames = np.frombuffer(pcm[: len(pcm) // 4 * 4], dtype="<i2").reshape(-1, 2)
    if agent_channel == "left":
        return frames[:, 1], frames[:, 0], sample_rate
    return frames[:, 0], frames[:, 1], sample_rate


def _recording_bytes(audio: Optional[Audio]) -> Optional[bytes]:
    if audio is None:
        return None
    try:
        return audio.get_bytes()
    except Exception:
        return None


def read_call_recording(
    audio: Optional[Audio], agent_channel: str = "right"
) -> Optional[Tuple[Any, Any, int]]:
    """The caller and agent channels of a 16-bit stereo WAV, and its rate."""
    wav = _recording_bytes(audio)
    return None if wav is None else _read_channels(wav, agent_channel)


def _recording_activity(
    wav: bytes, use_speech_detection: bool, agent_channel: str
) -> Optional[CallActivity]:
    channels = _read_channels(wav, agent_channel)
    if channels is None:
        return None
    caller_samples, agent_samples, sample_rate = channels
    agent_levels = _frame_levels(agent_samples, sample_rate)
    _, noise_level = speech_and_noise_levels(agent_levels)
    agent = _channel_spans(_loud_frames(agent_levels))
    stops = []
    for span in agent:
        begin = max(span.start, span.end - STOP_CONTEXT_SECONDS)
        sound = agent_samples[
            int(begin * sample_rate) : int(
                (span.end + STOP_MARGIN_SECONDS) * sample_rate
            )
        ]
        stops.append(
            SoundStop(span.end, ends_abruptly(sound, sample_rate, noise_level))
        )
    caller_levels = _frame_levels(caller_samples, sample_rate)
    caller_talking = (
        speech_frames(caller_samples, sample_rate, caller_levels)
        if use_speech_detection
        else _loud_frames(caller_levels)
    )
    return CallActivity(
        source=CALL_RECORDING,
        agent=tuple(agent),
        caller=tuple(_channel_spans(caller_talking)),
        duration=len(agent_samples) / sample_rate,
        agent_stops=tuple(stops),
    )


class _ContentCache:
    """Readings keyed by the audio's content, so metrics handed separate
    copies of the same call share one reading, and metrics asking at once
    wait for it."""

    def __init__(self, size: int):
        self._size = size
        self._readings: "OrderedDict[Tuple[str, Hashable], Any]" = OrderedDict()
        self._lock = threading.Lock()
        self._key_locks: Dict[Tuple[str, Hashable], threading.Lock] = {}

    def _found(self, key: Tuple[str, Hashable]) -> Tuple[bool, Any]:
        with self._lock:
            if key in self._readings:
                self._readings.move_to_end(key)
                return True, self._readings[key]
            return False, None

    def read(
        self,
        content: bytes,
        variant: Hashable,
        reader: Callable[[], T],
    ) -> T:
        """`variant` is whatever else changes the reading, such as whether
        speech detection is on."""
        key = (hashlib.sha256(content).hexdigest(), variant)
        found, reading = self._found(key)
        if found:
            return reading
        with self._lock:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            found, reading = self._found(key)
            if found:
                return reading
            reading = reader()
            with self._lock:
                self._readings[key] = reading
                while len(self._readings) > self._size:
                    self._readings.popitem(last=False)
                self._key_locks.pop(key, None)
        return reading

    def clear(self) -> None:
        with self._lock:
            self._readings.clear()


_recording_cache = _ContentCache(RECORDING_CACHE_SIZE)
_turn_audio_cache = _ContentCache(TURN_AUDIO_CACHE_SIZE)


def recording_activity(
    audio: Optional[Audio],
    use_speech_detection: bool,
    agent_channel: str = "right",
) -> Optional[CallActivity]:
    """None unless `audio` is a readable 16-bit stereo WAV."""
    wav = _recording_bytes(audio)
    if wav is None:
        return None
    return _recording_cache.read(
        wav,
        (use_speech_detection, agent_channel),
        lambda: _recording_activity(wav, use_speech_detection, agent_channel),
    )


def call_recording_activity(
    test_case: ConversationalTestCase, use_speech_detection: bool
) -> Optional[CallActivity]:
    return recording_activity(
        test_case.audio, use_speech_detection, test_case.agent_channel
    )


def _read_turn_audio(
    audio: Audio, use_speech_detection: bool
) -> Tuple[Tuple[float, float], ...]:
    try:
        samples, sample_rate = decode_audio(audio)
    except (TypeError, ValueError):
        return ()
    levels = _frame_levels(samples, sample_rate)
    if use_speech_detection:
        talking = speech_frames(
            samples, sample_rate, levels, min_level=TURN_SPEECH_MIN_LEVEL
        )
        spans = _channel_spans(talking, TURN_SPEECH_MIN_SECONDS)
    else:
        spans = _channel_spans(_loud_frames(levels))
    return tuple((span.start, span.end) for span in spans)


def _speech_in(
    entry: AudioTimelineEntry, use_speech_detection: bool
) -> List[Span]:
    """Where the speaker is heard in one turn's audio, on the call's clock.
    The whole audio counts when nothing is heard or it can't be decoded."""
    try:
        content = entry.audio.get_bytes()
    except Exception:
        content = None
    heard = (
        ()
        if content is None
        else _turn_audio_cache.read(
            content,
            use_speech_detection,
            lambda: _read_turn_audio(entry.audio, use_speech_detection),
        )
    )
    if not heard:
        return [Span(entry.start_time, entry.end_time, entry.turn_index)]
    return [
        Span(
            entry.start_time + start,
            min(entry.start_time + end, entry.end_time),
            entry.turn_index,
        )
        for start, end in heard
    ]


def _heard(
    timeline: Sequence[AudioTimelineEntry], use_speech_detection: bool
) -> Dict[str, List[Span]]:
    spans: Dict[str, List[Span]] = {"assistant": [], "user": []}
    for entry in timeline:
        if entry.role in spans:
            spans[entry.role].extend(_speech_in(entry, use_speech_detection))
    return {
        role: sorted(heard, key=lambda span: span.start)
        for role, heard in spans.items()
    }


def _speech_by_turn(
    heard: Dict[str, List[Span]],
) -> Dict[int, Tuple[float, float]]:
    speech: Dict[int, Tuple[float, float]] = {}
    for spans in heard.values():
        for span in spans:
            if span.turn is None:
                continue
            start, end = speech.get(span.turn, (span.start, span.end))
            speech[span.turn] = (min(start, span.start), max(end, span.end))
    return speech


def turn_speech(
    turns: Sequence[Turn], use_speech_detection: bool
) -> Dict[int, Tuple[float, float]]:
    """Where speech starts and ends in each timed turn, on the call's clock."""
    return _speech_by_turn(
        _heard(build_audio_timeline(turns), use_speech_detection)
    )


def turn_activity(
    test_case: ConversationalTestCase, use_speech_detection: bool
) -> CallActivity:
    timeline, skipped, untimed_reason = call_timeline(test_case)
    heard = _heard(timeline, use_speech_detection)
    return CallActivity(
        source=TURN_TIMESTAMPS,
        agent=tuple(heard["assistant"]),
        caller=tuple(heard["user"]),
        duration=max((entry.end_time for entry in timeline), default=0.0),
        skipped=tuple(skipped),
        untimed_reason=untimed_reason,
    )


def call_activity(
    test_case: ConversationalTestCase, use_speech_detection: bool
) -> CallActivity:
    return call_recording_activity(
        test_case, use_speech_detection
    ) or turn_activity(test_case, use_speech_detection)


def _speech_runs_across(
    caller: AudioTimelineEntry,
    reply: AudioTimelineEntry,
    caller_end: float,
    reply_start: float,
) -> bool:
    """The two turns' audio touches and speech runs to both edges, so no
    silence between them can be heard."""
    return (
        abs(reply.start_time - caller.end_time)
        <= CONTIGUOUS_CLIP_TOLERANCE_SECONDS
        and caller.end_time - caller_end <= SPEECH_AT_EDGE_SECONDS
        and reply_start - reply.start_time <= SPEECH_AT_EDGE_SECONDS
    )


def agent_reply_gaps(
    test_case: ConversationalTestCase, use_speech_detection: bool
) -> Tuple[Dict[int, float], List[Dict], Optional[str]]:
    """Seconds from where the caller's speech ends to where the agent's reply
    starts: the reply's recorded latency when it has one, else measured from
    the speech in each turn's audio, negative when the agent started before
    the caller finished."""
    timeline, skipped, untimed_reason = call_timeline(test_case)
    timed = {entry.turn_index for entry in timeline}
    speech = _speech_by_turn(_heard(timeline, use_speech_detection))
    gaps: Dict[int, float] = {}
    for position, reply in enumerate(timeline):
        if reply.role != "assistant":
            continue
        caller_index = reply.turn_index - 1
        if (
            caller_index >= 0
            and caller_index not in timed
            and test_case.turns[caller_index].role == "user"
            and test_case.turns[caller_index].audio is not None
        ):
            skipped.append(
                {
                    "turn": reply.turn_index,
                    "error": "The caller's audio before this reply could not "
                    "be decoded to time it.",
                }
            )
            continue
        previous = timeline[position - 1] if position else None
        if previous is None or previous.role != "user":
            continue
        latency_ms = test_case.turns[reply.turn_index].latency_ms
        if latency_ms is not None:
            gaps[reply.turn_index] = latency_ms / 1000
            continue
        caller_end = speech[previous.turn_index][1]
        reply_start = speech[reply.turn_index][0]
        if _speech_runs_across(previous, reply, caller_end, reply_start):
            skipped.append(
                {
                    "turn": reply.turn_index,
                    "error": "Reply audio touches the caller's with no "
                    "gap and no recorded latency.",
                }
            )
            continue
        gaps[reply.turn_index] = reply_start - caller_end
    return gaps, skipped, untimed_reason


def agent_reply_delays(
    test_case: ConversationalTestCase, use_speech_detection: bool
) -> Tuple[Dict[int, float], List[Dict], Optional[str]]:
    """How long the agent waited before each reply. Replies that talked over
    the caller had no wait and are left to Agent Interruption."""
    gaps, skipped, untimed_reason = agent_reply_gaps(
        test_case, use_speech_detection
    )
    delays: Dict[int, float] = {}
    for turn, gap in gaps.items():
        if gap < -AGENT_OVERLAP_ALLOWANCE_SECONDS:
            skipped.append(
                {
                    "turn": turn,
                    "error": "Agent started before the caller finished.",
                }
            )
            continue
        delays[turn] = max(gap, 0.0)
    return delays, skipped, untimed_reason


def reply_delays(
    test_case: ConversationalTestCase, use_speech_detection: bool
) -> Tuple[Dict, List[Dict], Optional[str], str]:
    """How long the agent waited before each reply, the replies left out, why
    the call could not be timed, and where the timing came from. Replies that
    started over the caller had no wait and are left to Agent Interruption."""
    recording = call_recording_activity(test_case, use_speech_detection)
    if recording is None:
        delays, skipped, untimed_reason = agent_reply_delays(
            test_case, use_speech_detection
        )
        return delays, skipped, untimed_reason, TURN_TIMESTAMPS
    delays: Dict[float, float] = {}
    skipped: List[Dict] = []
    for start in agent_starts(recording):
        if not start.caller_spoke_last:
            continue
        at = round(start.at, 2)
        if start.caller_runs_past > AGENT_OVERLAP_ALLOWANCE_SECONDS:
            skipped.append(
                {
                    "reply_start": at,
                    "error": "Agent started before the caller finished.",
                }
            )
            continue
        delays[at] = start.caller_silent_for
    return delays, skipped, None, CALL_RECORDING


def wait_span(
    test_case: ConversationalTestCase,
    event_key: str,
    key: Any,
    wait: float,
    use_speech_detection: bool,
) -> Dict:
    """The silence the caller sat through before a reply, on the call's
    clock: it ends where the reply's speech starts."""
    if event_key == EVENT_KEYS[CALL_RECORDING]:
        reply_start = key
    else:
        speech = turn_speech(test_case.turns, use_speech_detection).get(key)
        reply_start = None if speech is None else speech[0]
    if reply_start is None:
        return {}
    return on_call(reply_start - wait, wait)
