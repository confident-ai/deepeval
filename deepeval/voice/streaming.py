"""Bridging a speech stream into a connector's uplink.

Synthesis can hand back an utterance in frames as it is produced. Uplinks
consume those frames at different granularities — some forward each one, some
need the utterance whole — but every caller still needs the finished utterance
to record what was said, so these helpers assemble one while the frames go out.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterable, List, Optional, Tuple

from deepeval.test_case import Audio, AudioChunk
from deepeval.voice.connectors import audio_utils

DEFAULT_STREAM_SAMPLE_RATE = 24000

__all__ = [
    "DEFAULT_STREAM_SAMPLE_RATE",
    "PcmRecorder",
    "RealTimePacer",
    "UplinkResult",
    "collect_pcm_chunks",
    "pcm_to_audio",
]


@dataclass
class UplinkResult:
    """What was said, and when the agent first had any of it.

    Only the transport knows when speech left for the agent, and the answer
    differs by transport: one that forwards frames sends the first one while
    the rest is still being synthesized, while one that needs the utterance
    whole cannot send anything until synthesis ends. Timing a clip from
    synthesis instead would place the caller's voice on the call before the
    agent could have heard a word of it.
    """

    audio: Audio
    first_frame_at: Optional[float] = None


class PcmRecorder:
    """Keeps the utterance as its frames stream past."""

    def __init__(self, sample_rate: Optional[int] = None):
        self._parts: List[bytes] = []
        self.sample_rate = sample_rate

    def add(self, chunk: AudioChunk) -> bytes:
        """Record one frame's PCM and hand it back for sending."""
        if chunk.sampleRate:
            self.sample_rate = chunk.sampleRate
        pcm = chunk.get_bytes()
        self._parts.append(pcm)
        return pcm

    @property
    def pcm(self) -> bytes:
        return b"".join(self._parts)

    def to_audio(self) -> Audio:
        return pcm_to_audio(self.pcm, self.sample_rate)


class RealTimePacer:
    """Holds an uplink to the speed the caller could actually talk.

    A socket takes a whole utterance in milliseconds, so an agent handed one
    that way answers before the caller could have finished saying it: the
    recording then shows the reply starting inside the caller's own turn, and
    the wait before it comes out negative. Worse, an agent listening for a
    pause hears the trailing silence almost at once and cuts in early. Frames
    leave at the rate they would be spoken instead, which is what any real
    line does and what makes the timings mean anything.

    Nothing is done about speech that cannot be produced fast enough — that
    gap is real, and the caller does audibly stall.
    """

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self._started_at: Optional[float] = None
        self._sent_seconds = 0.0

    async def wait_to_send(self, pcm: bytes) -> None:
        """Sleep until this audio is due to be heard, then account for it."""
        now = time.perf_counter()
        if self._started_at is None:
            self._started_at = now
        ahead = self._sent_seconds - (now - self._started_at)
        if ahead > 0:
            await asyncio.sleep(ahead)
        self._sent_seconds += (len(pcm) / 2) / self.sample_rate


async def collect_pcm_chunks(
    chunks: AsyncIterable[AudioChunk],
) -> Tuple[bytes, int]:
    """Drain a speech stream into a single PCM buffer and its sample rate."""
    recorder = PcmRecorder()
    async for chunk in chunks:
        recorder.add(chunk)
    return recorder.pcm, recorder.sample_rate or DEFAULT_STREAM_SAMPLE_RATE


def pcm_to_audio(pcm: bytes, sample_rate: Optional[int] = None) -> Audio:
    rate = sample_rate or DEFAULT_STREAM_SAMPLE_RATE
    return Audio.from_bytes(
        audio_utils.pcm16_to_wav_bytes(pcm, rate, 1),
        "audio/wav",
        sampleRate=rate,
        encoding="wav",
        duration=(len(pcm) / 2) / rate if rate else None,
    )
