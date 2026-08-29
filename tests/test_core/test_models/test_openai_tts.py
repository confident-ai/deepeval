"""Streaming synthesis on `OpenAITTSModel`."""

from typing import List

import pytest

from deepeval.models.tts_models.openai_tts import (
    _STREAM_FRAME_BYTES,
    OpenAITTSModel,
)

_SAMPLE_RATE = 24000


class _FakeStreamedResponse:
    def __init__(self, reads: List[bytes]):
        self._reads = reads

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def iter_bytes(self, *args, **kwargs):
        for read in self._reads:
            yield read


class _FakeSpeech:
    def __init__(self, reads: List[bytes]):
        self._reads = reads
        self.calls: List[dict] = []

    @property
    def with_streaming_response(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeStreamedResponse(self._reads)


class _FakeClient:
    def __init__(self, speech: _FakeSpeech):
        self.audio = type("_Audio", (), {"speech": speech})()


def _model(reads: List[bytes], **kwargs) -> tuple:
    model = OpenAITTSModel(api_key="test-key", **kwargs)
    speech = _FakeSpeech(reads)
    model._async_client = _FakeClient(speech)
    return model, speech


async def _collect(model, text: str = "Hello there.", **kwargs) -> List:
    return [chunk async for chunk in model.a_synthesize_stream(text, **kwargs)]


@pytest.mark.asyncio
async def test_stream_yields_fixed_frames_with_a_final_flag():
    # Three and a half frames, arriving in reads that straddle frame edges.
    audio = bytes(range(256)) * 66  # 16,896 bytes
    reads = [audio[i : i + 1000] for i in range(0, len(audio), 1000)]
    model, _ = _model(reads)

    chunks = await _collect(model)

    sizes = [len(chunk.get_bytes()) for chunk in chunks]
    remainder = len(audio) % _STREAM_FRAME_BYTES
    assert sizes == [_STREAM_FRAME_BYTES] * 3 + [remainder]
    assert [chunk.final for chunk in chunks] == [False, False, False, True]
    assert b"".join(chunk.get_bytes() for chunk in chunks) == audio


@pytest.mark.asyncio
async def test_stream_describes_its_frames_as_pcm():
    model, _ = _model([b"\x01\x02" * 100])

    (chunk,) = await _collect(model)

    assert chunk.mimeType == "audio/pcm"
    assert chunk.encoding == "pcm"
    assert chunk.sampleRate == _SAMPLE_RATE
    assert chunk.duration == pytest.approx(200 / 2 / _SAMPLE_RATE)
    assert chunk.final is True


@pytest.mark.asyncio
async def test_stream_requests_pcm_even_when_configured_for_a_container():
    model, speech = _model([b"\x00\x00" * 10], response_format="mp3")

    await _collect(model, voice="coral")

    assert speech.calls[0]["response_format"] == "pcm"
    assert speech.calls[0]["voice"] == "coral"


@pytest.mark.asyncio
async def test_an_exact_multiple_of_a_frame_still_ends_with_a_final_chunk():
    model, _ = _model([b"\x07\x00" * _STREAM_FRAME_BYTES])

    chunks = await _collect(model)

    assert [len(c.get_bytes()) for c in chunks] == [_STREAM_FRAME_BYTES] * 2
    assert [c.final for c in chunks] == [False, True]


@pytest.mark.asyncio
async def test_streaming_is_advertised():
    model, _ = _model([])

    assert model.supports_streaming() is True
