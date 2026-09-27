"""Build a synchronized call timeline from audio attached to turns."""

from array import array
from dataclasses import dataclass
from typing import List, Optional, Sequence

from deepeval.test_case import Audio, Turn
from deepeval.voice.connectors import audio_utils


@dataclass(frozen=True)
class AudioTimelineEntry:
    turn_index: int
    role: str
    start_time: float
    duration: float
    audio: Audio

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


def audio_duration(audio: Audio) -> Optional[float]:
    """Return a trustworthy duration, deriving WAV duration when possible."""
    try:
        pcm, sample_rate, channels = audio_utils.wav_bytes_to_pcm16(
            audio.get_bytes()
        )
    except Exception:
        return audio.duration
    if not sample_rate:
        return audio.duration
    duration = (len(pcm) / 2 / max(channels, 1)) / sample_rate
    audio.duration = duration
    return duration


def build_audio_timeline(
    turns: Sequence[Turn],
    *,
    require_start_times: bool = True,
) -> List[AudioTimelineEntry]:
    """Place turn audio on a call-relative timeline.

    When ``require_start_times`` is false, legacy clips without timestamps are
    placed sequentially after the preceding clip. Metrics should keep the
    default so they never invent timing evidence.
    """
    entries: List[AudioTimelineEntry] = []
    sequential_cursor = 0.0
    for index, turn in enumerate(turns):
        audio = turn.audio
        if audio is None:
            continue
        duration = audio_duration(audio)
        if duration is None:
            continue
        if audio.start_time is None:
            if require_start_times:
                continue
            start_time = sequential_cursor
        else:
            start_time = audio.start_time
        entries.append(
            AudioTimelineEntry(
                turn_index=index,
                role=turn.role,
                start_time=start_time,
                duration=duration,
                audio=audio,
            )
        )
        sequential_cursor = max(sequential_cursor, start_time + duration)
    return sorted(
        entries, key=lambda entry: (entry.start_time, entry.turn_index)
    )


def render_timeline_wav(
    turns: Sequence[Turn],
    *,
    require_start_times: bool = True,
) -> Optional[bytes]:
    """Render timestamped WAV clips onto one mono WAV, preserving overlap.

    The recording starts at the first clip: time before anyone speaks is
    simulator setup (connecting, writing and synthesizing the opening line),
    not part of the conversation. Offsets *between* clips are preserved — the
    turns keep their absolute ``start_time`` for metrics.
    """
    entries = build_audio_timeline(
        turns, require_start_times=require_start_times
    )
    if not entries:
        return None
    origin = entries[0].start_time

    target_rate: Optional[int] = None
    decoded = []
    for entry in entries:
        try:
            pcm, rate, channels = audio_utils.wav_bytes_to_pcm16(
                entry.audio.get_bytes()
            )
        except Exception:
            return None
        pcm = audio_utils.downmix_to_mono(pcm, channels or 1)
        if target_rate is None:
            target_rate = rate
        elif rate != target_rate:
            pcm = audio_utils.resample_pcm16(pcm, rate, target_rate)
        samples = array("h")
        samples.frombytes(pcm)
        decoded.append((entry, samples))

    if not target_rate:
        return None
    total_samples = max(
        int(round((entry.end_time - origin) * target_rate))
        for entry, _ in decoded
    )
    mixed = array("i", [0]) * total_samples
    for entry, samples in decoded:
        offset = int(round((entry.start_time - origin) * target_rate))
        needed = offset + len(samples)
        if needed > len(mixed):
            mixed.extend([0] * (needed - len(mixed)))
        for sample_index, sample in enumerate(samples):
            mixed[offset + sample_index] += sample

    output = array(
        "h",
        (max(-32768, min(32767, sample)) for sample in mixed),
    )
    return audio_utils.pcm16_to_wav_bytes(output.tobytes(), target_rate, 1)
