"""The HTTP-based TTS providers: framing, request shape and cost."""

import json as jsonlib
from typing import List

import pytest

from deepeval.errors import DeepEvalError
from deepeval.models import (
    CartesiaTTSModel,
    DeepgramTTSModel,
    ElevenLabsTTSModel,
)
from deepeval.models.tts._frames import frame_size_bytes
from deepeval.models.tts.deepgram.deepgram import apply_voice
from tests.test_core.test_models.speech_stubs import FakeTransport

_RATE = 24000
_FRAME = frame_size_bytes(_RATE)


def _elevenlabs(**kwargs):
    model = ElevenLabsTTSModel(api_key="test-key", **kwargs)
    return model


def _deepgram(**kwargs):
    return DeepgramTTSModel(api_key="test-key", **kwargs)


def _cartesia(**kwargs):
    return CartesiaTTSModel(api_key="test-key", voice="voice-id", **kwargs)


def _with_transport(model, **transport_kwargs):
    transport = FakeTransport(**transport_kwargs)
    model.model = transport
    return model, transport


async def _collect(model, text: str = "Hello there.", **kwargs) -> List:
    return [chunk async for chunk in model.a_synthesize_stream(text, **kwargs)]


#
# Framing (shared by all three providers)
#


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory", [_elevenlabs, _deepgram, _cartesia], ids=["11labs", "dg", "ct"]
)
async def test_stream_yields_fixed_frames_with_a_final_flag(factory):
    # Three and a half frames, arriving in reads that straddle frame edges.
    audio = bytes(range(256)) * 66
    reads = [audio[i : i + 1000] for i in range(0, len(audio), 1000)]
    model, _ = _with_transport(factory(), stream_reads=reads)

    chunks = await _collect(model)

    sizes = [len(chunk.get_bytes()) for chunk in chunks]
    assert sizes == [_FRAME] * (len(audio) // _FRAME) + [len(audio) % _FRAME]
    assert [chunk.final for chunk in chunks] == [False, False, False, True]
    assert b"".join(chunk.get_bytes() for chunk in chunks) == audio


@pytest.mark.asyncio
async def test_an_exact_multiple_of_a_frame_still_ends_with_a_final_chunk():
    model, _ = _with_transport(
        _elevenlabs(), stream_reads=[b"\x07\x00" * _FRAME]
    )

    chunks = await _collect(model)

    assert [len(c.get_bytes()) for c in chunks] == [_FRAME] * 2
    assert [c.final for c in chunks] == [False, True]


@pytest.mark.asyncio
async def test_stream_describes_its_frames_as_pcm():
    model, _ = _with_transport(_elevenlabs(), stream_reads=[b"\x01\x02" * 100])

    (chunk,) = await _collect(model)

    assert chunk.mimeType == "audio/pcm"
    assert chunk.encoding == "pcm"
    assert chunk.sampleRate == _RATE
    assert chunk.duration == pytest.approx(200 / 2 / _RATE)
    assert chunk.final is True


@pytest.mark.asyncio
async def test_a_silent_stream_yields_nothing():
    model, _ = _with_transport(_elevenlabs(), stream_reads=[])

    assert await _collect(model) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory", [_elevenlabs, _deepgram, _cartesia], ids=["11labs", "dg", "ct"]
)
async def test_streaming_is_advertised(factory):
    assert factory().supports_streaming() is True


#
# ElevenLabs
#


@pytest.mark.asyncio
async def test_elevenlabs_batch_asks_for_a_wav_and_streams_pcm():
    model, transport = _with_transport(
        _elevenlabs(), content=b"RIFF", stream_reads=[b"\x00\x00"]
    )

    audio, _ = await model.a_synthesize("Hi")
    assert transport.params_of()["output_format"] == f"wav_{_RATE}"
    assert audio.mimeType == "audio/wav"
    assert audio.sampleRate == _RATE

    await _collect(model)
    # Streaming cannot use a container: its header declares a length the first
    # frame does not know yet.
    assert transport.params_of()["output_format"] == f"pcm_{_RATE}"


@pytest.mark.asyncio
async def test_elevenlabs_puts_the_voice_in_the_path_and_the_model_in_the_body():
    model, transport = _with_transport(_elevenlabs(), content=b"RIFF")

    await model.a_synthesize("Hi", voice="custom-voice")

    assert transport.last["path"] == "/v1/text-to-speech/custom-voice"
    assert transport.last["json"] == {
        "text": "Hi",
        "model_id": "eleven_flash_v2_5",
    }


@pytest.mark.asyncio
async def test_elevenlabs_streams_from_the_stream_endpoint():
    model, transport = _with_transport(
        _elevenlabs(), stream_reads=[b"\x00\x00"]
    )

    await _collect(model)

    assert transport.last["path"].endswith("/stream")


def test_elevenlabs_does_not_default_to_a_sunsetting_voice():
    # The legacy "Default" voices (Rachel, Adam, Bella, ...) stop resolving on
    # 2027-01-01 and already 404 for accounts created after March 2026.
    retired = {
        "21m00Tcm4TlvDq8ikWAM",
        "pNInz6obpgDQGcFmaJgB",
        "EXAVITQu4vr4xnSDxMaL",
        "TxGEqnHWrfWFTfGW9XjX",
        "yoZ06aMxZJJ28mfd3POQ",
        "ErXwobaYiN019PkySvjV",
    }
    assert _elevenlabs().voice not in retired


def test_elevenlabs_rejects_a_sample_rate_it_cannot_emit():
    with pytest.raises(ValueError, match="16000"):
        _elevenlabs(sample_rate=11111)


def test_elevenlabs_prices_flash_below_the_expressive_models():
    text = "a" * 1_000_000
    assert _elevenlabs().synthesis_cost(text) == pytest.approx(50.0)
    assert _elevenlabs(model="eleven_v3").synthesis_cost(text) == pytest.approx(
        100.0
    )
    assert _elevenlabs(cost_per_1m_chars=7.5).synthesis_cost(
        text
    ) == pytest.approx(7.5)


def test_an_unknown_elevenlabs_model_has_no_price_rather_than_a_wrong_one():
    assert (
        _elevenlabs(model="eleven_something_new").synthesis_cost("hi") is None
    )


#
# Deepgram
#


@pytest.mark.asyncio
async def test_deepgram_sends_text_in_the_body_and_everything_else_in_the_query():
    model, transport = _with_transport(_deepgram(), content=b"RIFF")

    await model.a_synthesize("Hi")

    assert transport.last["path"] == "/v1/speak"
    assert transport.last["json"] == {"text": "Hi"}
    assert transport.params_of() == {
        "model": "aura-2-thalia-en",
        "encoding": "linear16",
        "container": "wav",
        "sample_rate": str(_RATE),
    }


@pytest.mark.asyncio
async def test_deepgram_drops_the_container_when_streaming():
    model, transport = _with_transport(_deepgram(), stream_reads=[b"\x00\x00"])

    await _collect(model)

    assert transport.params_of()["container"] == "none"


@pytest.mark.parametrize(
    "model_name,voice,expected",
    [
        ("aura-2-thalia-en", "zeus", "aura-2-zeus-en"),
        ("aura-2-thalia-en", "aura-2-luna-es", "aura-2-luna-es"),
        ("aura-asteria-en", "orion", "aura-orion-en"),
        ("aura-2-thalia-ja", "fujin", "aura-2-fujin-ja"),
    ],
)
def test_deepgram_rewrites_the_voice_segment_of_the_model_name(
    model_name, voice, expected
):
    # An Aura model name carries voice and language, so selecting a voice means
    # editing the identifier rather than sending a separate field.
    assert apply_voice(model_name, voice) == expected


def test_deepgram_folds_a_constructor_voice_into_the_reported_model_name():
    assert _deepgram(voice="zeus").get_model_name() == "aura-2-zeus-en"


def test_deepgram_prices_aura_2_above_aura_1():
    text = "a" * 1_000_000
    assert _deepgram().synthesis_cost(text) == pytest.approx(30.0)
    assert _deepgram(model="aura-asteria-en").synthesis_cost(
        text
    ) == pytest.approx(15.0)


#
# Cartesia
#


@pytest.mark.asyncio
async def test_cartesia_nests_the_output_format_and_switches_container():
    model, transport = _with_transport(
        _cartesia(), content=b"RIFF", stream_reads=[b"\x00\x00"]
    )

    await model.a_synthesize("Hi")
    body = transport.last["json"]
    assert body["model_id"] == "sonic-3.6"
    assert body["transcript"] == "Hi"
    assert body["voice"] == "voice-id"
    assert body["output_format"] == {
        "container": "wav",
        "encoding": "pcm_s16le",
        "sample_rate": _RATE,
    }

    await _collect(model)
    assert transport.last["json"]["output_format"]["container"] == "raw"


def test_cartesia_does_not_default_to_a_retired_model():
    # sonic-2 and sonic-turbo have left the model enum and now fail with
    # model_not_found.
    assert _cartesia().get_model_name() not in {"sonic-2", "sonic-turbo"}


def test_cartesia_pins_the_dated_api_version_it_was_written_against():
    model = _cartesia()
    assert model.model.headers["Cartesia-Version"] == "2026-08-14"


def test_cartesia_requires_a_voice():
    model = CartesiaTTSModel(api_key="test-key")
    model.model = FakeTransport(content=b"RIFF")

    with pytest.raises(DeepEvalError, match="no default voice"):
        model.synthesize("Hi")


def test_cartesia_reports_no_cost_without_an_explicit_rate():
    # Cartesia bills in credits and publishes no dollar conversion, so a
    # default price would be invented.
    assert _cartesia().synthesis_cost("a" * 1000) is None
    assert _cartesia(cost_per_1m_chars=25.0).synthesis_cost(
        "a" * 1_000_000
    ) == pytest.approx(25.0)


#
# Sync parity
#


@pytest.mark.parametrize(
    "factory", [_elevenlabs, _deepgram, _cartesia], ids=["11labs", "dg", "ct"]
)
def test_the_sync_path_requests_the_same_thing_as_the_async_one(factory):
    import asyncio

    sync_model, sync_transport = _with_transport(factory(), content=b"RIFF")
    sync_model.synthesize("Hi")

    async_model, async_transport = _with_transport(factory(), content=b"RIFF")
    asyncio.run(async_model.a_synthesize("Hi"))

    assert jsonlib.dumps(
        sync_transport.last, sort_keys=True, default=str
    ) == jsonlib.dumps(async_transport.last, sort_keys=True, default=str)
