"""When each speaker made sound on a call. With a stereo recording of the
whole call (caller left, agent right) it comes from each channel's loudness;
without one, from the turns' timestamps."""

import functools
import os
import wave
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from deepeval.metrics.voice.utils.analysis import (
    STOP_CONTEXT_SECONDS,
    speech_and_noise_levels,
    ends_abruptly,
    frame_samples,
    get_numpy,
    rms,
)
from deepeval.metrics.voice.utils.defects import (
    AGENT_OVERLAP_ALLOWANCE_SECONDS,
    CALL_RECORDING,
    TURN_TIMESTAMPS,
    agent_reply_delays,
    call_timeline,
)
from deepeval.metrics.voice.utils.events import on_call
from deepeval.test_case import ConversationalTestCase
from deepeval.voice.connectors import audio_utils

EVENT_KEYS = {CALL_RECORDING: "reply_start", TURN_TIMESTAMPS: "turn"}
# Gaps shorter than this sit between syllables, not between utterances.
SOUND_GAP_SECONDS = 0.3
# Shorter bursts are clicks and breaths.
MIN_SOUND_SECONDS = 0.15
# A pause shorter than this doesn't end a stretch of the agent's speech.
AGENT_PAUSE_SECONDS = 1.0
# A speaker takes the floor after at least this much of their own silence.
SPEAKER_PAUSE_SECONDS = 0.5
# The caller is already talking when this much of the moment before the agent
# starts holds caller sound.
CALLER_TALKING_WINDOW_SECONDS = 0.3
CALLER_TALKING_MIN_SECONDS = 0.2
# Room past the last 20 ms frame to find where the sound really stops.
STOP_MARGIN_SECONDS = 0.02


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


def _caller_pause(
    caller_stretches: Sequence[Span], at: float
) -> Optional[float]:
    """The caller's pause around `at`, from when they stopped speaking to when
    they started again; 0 when they had already been talking."""
    for index, stretch in enumerate(caller_stretches):
        if stretch.end <= at:
            continue
        if stretch.start < at - CALLER_TALKING_WINDOW_SECONDS:
            return 0.0
        if index == 0:
            return None
        return stretch.start - caller_stretches[index - 1].end
    return None


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
        caller_next = next(
            (max(at, span.start) for span in caller_stretches if span.end > at),
            None,
        )
        starts.append(
            AgentStart(
                at=at,
                stops_at=stretch.end,
                turn=stretch.turn,
                caller_talking=sound_between(
                    caller, at - CALLER_TALKING_WINDOW_SECONDS, at
                )
                >= CALLER_TALKING_MIN_SECONDS,
                caller_starts_after=(
                    None if caller_next is None else caller_next - at
                ),
                caller_silent_for=(
                    None if caller_end is None else max(0.0, at - caller_end)
                ),
                caller_pause_seconds=_caller_pause(caller_stretches, at),
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


def _channel_spans(levels: Sequence[float]) -> List[Span]:
    frame_seconds = audio_utils.DEFAULT_FRAME_MS / 1000
    bounds: List[List[float]] = []
    for index, level in enumerate(levels):
        if level < audio_utils.DEFAULT_SILENCE_RMS:
            continue
        start = index * frame_seconds
        if bounds and start - bounds[-1][1] < SOUND_GAP_SECONDS:
            bounds[-1][1] = start + frame_seconds
        else:
            bounds.append([start, start + frame_seconds])
    return [
        Span(start, end)
        for start, end in bounds
        if end - start >= MIN_SOUND_SECONDS
    ]


def _frame_levels(samples: Any, sample_rate: int) -> List[float]:
    return [
        rms(frame)
        for frame in frame_samples(
            samples, sample_rate, audio_utils.DEFAULT_FRAME_MS
        )
    ]


def _read_channels(path: str) -> Optional[Tuple[Any, Any, int]]:
    try:
        with open(path, "rb") as file:
            pcm, sample_rate, channels = audio_utils.wav_bytes_to_pcm16(
                file.read()
            )
    except (OSError, EOFError, ValueError, wave.Error):
        return None
    if channels != 2 or not sample_rate:
        return None
    np = get_numpy()
    frames = np.frombuffer(pcm[: len(pcm) // 4 * 4], dtype="<i2").reshape(-1, 2)
    return frames[:, 0], frames[:, 1], sample_rate


@functools.lru_cache(maxsize=4)
def _recording_activity(
    path: str, modified_ns: int, size: int
) -> Optional[CallActivity]:
    channels = _read_channels(path)
    if channels is None:
        return None
    caller_samples, agent_samples, sample_rate = channels
    agent_levels = _frame_levels(agent_samples, sample_rate)
    _, noise_level = speech_and_noise_levels(agent_levels)
    agent = _channel_spans(agent_levels)
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
    return CallActivity(
        source=CALL_RECORDING,
        agent=tuple(agent),
        caller=tuple(
            _channel_spans(_frame_levels(caller_samples, sample_rate))
        ),
        duration=len(agent_samples) / sample_rate,
        agent_stops=tuple(stops),
    )


def recording_activity(path: Optional[str]) -> Optional[CallActivity]:
    """None unless `path` is a readable 16-bit stereo WAV."""
    if not path:
        return None
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return _recording_activity(path, stat.st_mtime_ns, stat.st_size)


def call_recording_activity(
    test_case: ConversationalTestCase,
) -> Optional[CallActivity]:
    return recording_activity(test_case.call_recording_path)


def turn_activity(test_case: ConversationalTestCase) -> CallActivity:
    timeline, skipped, untimed_reason = call_timeline(test_case)
    spans = {
        role: tuple(
            Span(entry.start_time, entry.end_time, entry.turn_index)
            for entry in timeline
            if entry.role == role
        )
        for role in ("assistant", "user")
    }
    return CallActivity(
        source=TURN_TIMESTAMPS,
        agent=spans["assistant"],
        caller=spans["user"],
        duration=max((entry.end_time for entry in timeline), default=0.0),
        skipped=tuple(skipped),
        untimed_reason=untimed_reason,
    )


def call_activity(test_case: ConversationalTestCase) -> CallActivity:
    return call_recording_activity(test_case) or turn_activity(test_case)


def reply_delays(
    test_case: ConversationalTestCase,
) -> Tuple[Dict, List[Dict], Optional[str], str]:
    """How long the agent waited before each reply, the replies left out, why
    the call could not be timed, and where the timing came from. Replies that
    started over the caller had no wait and are left to Agent Interruption."""
    recording = call_recording_activity(test_case)
    if recording is None:
        delays, skipped, untimed_reason = agent_reply_delays(test_case)
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
    test_case: ConversationalTestCase, event_key: str, key: Any, wait: float
) -> Dict:
    """The silence the caller sat through before a reply, on the call's
    clock: it ends where the reply starts."""
    if event_key == EVENT_KEYS[CALL_RECORDING]:
        reply_start = key
    else:
        audio = test_case.turns[key].audio
        reply_start = None if audio is None else audio.start_time
    if reply_start is None:
        return {}
    return on_call(reply_start - wait, wait)
