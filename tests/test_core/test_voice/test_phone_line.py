from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from deepeval.test_case import AudioChunk
from deepeval.voice.phone_line import PhoneLine, PhoneLineTTS

SAMPLE_RATE = 24000
FRAME_BYTES = 1920 * 2


def _tone(frequency: float) -> bytes:
    times = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    return (
        (np.sin(2 * np.pi * frequency * times) * 8000)
        .astype(np.int16)
        .tobytes()
    )


def _rms(pcm: bytes) -> float:
    samples = np.frombuffer(pcm, np.int16).astype(float)[SAMPLE_RATE // 10 :]
    return float(np.sqrt(np.mean(samples**2)))


def test_the_phone_line_keeps_the_speech_band():
    speech = _tone(1000)

    assert len(PhoneLine(SAMPLE_RATE).process(speech)) == len(speech)
    assert _rms(PhoneLine(SAMPLE_RATE).process(speech)) > 0.6 * _rms(speech)


def test_the_phone_line_drops_hiss_and_hum():
    hiss, hum = _tone(8000), _tone(60)

    assert _rms(PhoneLine(SAMPLE_RATE).process(hiss)) < 0.1 * _rms(hiss)
    assert _rms(PhoneLine(SAMPLE_RATE).process(hum)) < 0.2 * _rms(hum)


def test_frames_join_exactly_like_one_continuous_signal():
    speech = _tone(1000)
    framed_line = PhoneLine(SAMPLE_RATE)

    framed = b"".join(
        framed_line.process(speech[start : start + FRAME_BYTES])
        for start in range(0, len(speech), FRAME_BYTES)
    )

    assert framed == PhoneLine(SAMPLE_RATE).process(speech)


@pytest.mark.asyncio
async def test_a_phone_line_tts_keeps_streaming():
    async def stream(*_args, **_kwargs):
        for index in range(3):
            yield AudioChunk.from_bytes(
                _tone(1000)[:FRAME_BYTES],
                "audio/pcm",
                sampleRate=SAMPLE_RATE,
                encoding="pcm",
                final=index == 2,
            )

    inner = MagicMock(sample_rate=SAMPLE_RATE)
    inner.get_model_name.return_value = "gpt-4o-mini-tts"
    inner.supports_streaming.return_value = True
    inner.a_synthesize_stream = stream
    inner.a_synthesize = AsyncMock()
    tts = PhoneLineTTS(inner)

    chunks = [
        chunk async for chunk in tts.a_synthesize_stream("Hi", voice="coral")
    ]

    assert tts.supports_streaming() is True
    assert [chunk.final for chunk in chunks] == [False, False, True]
    assert chunks[0].get_bytes() != _tone(1000)[:FRAME_BYTES]
    assert len(chunks[0].get_bytes()) == FRAME_BYTES
