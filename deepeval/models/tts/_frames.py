"""Turning a stream of PCM bytes into `AudioChunk`s."""

from __future__ import annotations

from typing import AsyncGenerator, AsyncIterable, Optional

from deepeval.test_case import AudioChunk

# A frame has to be playable on its own, which rules out every container format
# (their headers declare a length the first frame does not know yet), so a
# stream is always raw 16-bit mono PCM.
STREAM_MIME = "audio/pcm"
STREAM_ENCODING = "pcm"
DEFAULT_FRAME_SECONDS = 0.1


def pcm_chunk(
    data: bytes, *, sample_rate: int, final: bool = False
) -> AudioChunk:
    return AudioChunk.from_bytes(
        data,
        STREAM_MIME,
        sampleRate=sample_rate,
        encoding=STREAM_ENCODING,
        duration=len(data) / 2 / sample_rate,
        final=final,
    )


def frame_size_bytes(
    sample_rate: int, frame_seconds: float = DEFAULT_FRAME_SECONDS
) -> int:
    """Bytes in one frame, always even so a frame never splits a sample."""
    samples = max(1, int(sample_rate * frame_seconds))
    return samples * 2


async def frame_pcm_stream(
    source: AsyncIterable[bytes],
    *,
    sample_rate: int,
    frame_seconds: float = DEFAULT_FRAME_SECONDS,
) -> AsyncGenerator[AudioChunk, None]:
    """Re-cut arbitrary byte reads from an HTTP body into fixed audio frames.

    A provider's chunked response boundaries follow its network writes, not the
    audio, so reads arrive at unhelpful sizes and can split a sample in half.

    The last frame is held back so it can be flagged `final`, which is how a
    consumer knows the utterance is complete without also tracking the stream.
    """
    frame_bytes = frame_size_bytes(sample_rate, frame_seconds)
    buffer = bytearray()
    held: Optional[bytes] = None

    async for data in source:
        if not data:
            continue
        buffer.extend(data)
        while len(buffer) >= frame_bytes:
            frame = bytes(buffer[:frame_bytes])
            del buffer[:frame_bytes]
            if held is not None:
                yield pcm_chunk(held, sample_rate=sample_rate)
            held = frame

    if buffer:
        if held is not None:
            yield pcm_chunk(held, sample_rate=sample_rate)
        held = bytes(buffer)
    if held is not None:
        yield pcm_chunk(held, sample_rate=sample_rate, final=True)
