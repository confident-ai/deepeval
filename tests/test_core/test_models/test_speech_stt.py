"""The HTTP-based STT providers: transcript extraction, routing and cost."""

import json as jsonlib
from typing import List

import pytest

from deepeval.models import (
    AssemblyAISTTModel,
    CartesiaSTTModel,
    DeepgramSTTModel,
    ElevenLabsSTTModel,
    OpenAISTTModel,
)
from deepeval.models.speech import SpeechHTTPError
from deepeval.models.stt.assemblyai.assemblyai import SYNC_URL
from deepeval.test_case import AudioChunk
from tests.test_core.test_models.speech_stubs import FakeTransport, wav_audio


def _deepgram(**kwargs):
    return DeepgramSTTModel(api_key="test-key", **kwargs)


def _elevenlabs(**kwargs):
    return ElevenLabsSTTModel(api_key="test-key", **kwargs)


def _cartesia(**kwargs):
    return CartesiaSTTModel(api_key="test-key", **kwargs)


def _assemblyai(**kwargs):
    kwargs.setdefault("poll_interval_seconds", 0.0)
    return AssemblyAISTTModel(api_key="test-key", **kwargs)


def _with_transport(model, **transport_kwargs):
    transport = FakeTransport(**transport_kwargs)
    model.model = transport
    return model, transport


def _deepgram_payload(transcript: str):
    return {
        "results": {
            "channels": [{"alternatives": [{"transcript": transcript}]}]
        }
    }


#
# Truncated-audio padding
#


@pytest.mark.parametrize(
    "factory,expected",
    [
        # Whisper-derived and autoregressive: they finish a clipped word,
        # inventing speech the caller never heard.
        (_elevenlabs, 0.3),
        (_cartesia, 0.3),
        # These stop where the audio stops, so padding would only add silence.
        (_deepgram, 0.0),
        (_assemblyai, 0.0),
    ],
    ids=["11labs", "cartesia", "deepgram", "assemblyai"],
)
def test_padding_follows_whether_the_model_completes_clipped_words(
    factory, expected
):
    assert factory().truncated_audio_pad_seconds == expected


#
# Deepgram
#


@pytest.mark.asyncio
async def test_deepgram_reads_the_transcript_out_of_its_nested_response():
    model, transport = _with_transport(
        _deepgram(), json=_deepgram_payload("hello there")
    )

    text, _ = await model.a_transcribe(wav_audio())

    assert text == "hello there"
    assert transport.last["path"] == "/v1/listen"
    assert transport.params_of()["model"] == "nova-3"
    assert transport.params_of()["smart_format"] == "True"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"results": {"channels": []}},
        {"results": {"channels": [{"alternatives": []}]}},
        {"results": {}},
        {},
    ],
)
async def test_deepgram_treats_a_transcriptless_response_as_silence(payload):
    # Silence is a successful response with nothing in it, not an error.
    model, _ = _with_transport(_deepgram(), json=payload)

    text, _ = await model.a_transcribe(wav_audio())

    assert text == ""


@pytest.mark.asyncio
async def test_deepgram_turns_auto_into_explicit_language_detection():
    # Deepgram detects only when asked, so an omitted language means English,
    # not detection.
    model, transport = _with_transport(
        _deepgram(language="en"), json=_deepgram_payload("hola")
    )

    await model.a_transcribe(wav_audio(), language="auto")

    params = transport.params_of()
    assert params["detect_language"] == "True"
    assert "language" not in params


@pytest.mark.asyncio
async def test_deepgram_forwards_a_configured_language():
    model, transport = _with_transport(
        _deepgram(language="es"), json=_deepgram_payload("hola")
    )

    await model.a_transcribe(wav_audio())

    assert transport.params_of()["language"] == "es"


@pytest.mark.asyncio
async def test_deepgram_prices_by_the_minute_of_audio():
    model, _ = _with_transport(_deepgram(), json=_deepgram_payload("hi"))

    _, cost = await model.a_transcribe(wav_audio(seconds=30))

    assert cost == pytest.approx(0.5 * 0.0043)


#
# ElevenLabs
#


@pytest.mark.asyncio
async def test_elevenlabs_reads_a_flat_text_field():
    model, transport = _with_transport(
        _elevenlabs(), json={"text": "hello", "audio_duration_secs": 3600.0}
    )

    text, cost = await model.a_transcribe(wav_audio())

    assert text == "hello"
    assert transport.last["path"] == "/v1/speech-to-text"
    assert transport.last["multipart"].fields["model_id"] == "scribe_v2"
    # Priced per hour, and the response reports exactly one hour of audio.
    assert cost == pytest.approx(0.22)


@pytest.mark.asyncio
async def test_elevenlabs_joins_a_multi_channel_response():
    model, _ = _with_transport(
        _elevenlabs(),
        json={"transcripts": [{"text": "left"}, {"text": "right"}]},
    )

    text, _ = await model.a_transcribe(wav_audio())

    assert text == "left right"


@pytest.mark.asyncio
async def test_elevenlabs_omits_the_language_to_ask_for_detection():
    # Scribe detects whenever language_code is absent.
    model, transport = _with_transport(
        _elevenlabs(language="en"), json={"text": "hola"}
    )

    await model.a_transcribe(wav_audio(), language="auto")

    assert "language_code" not in transport.last["multipart"].fields


def test_elevenlabs_does_not_default_to_the_deprecated_scribe_v1():
    assert _elevenlabs().get_model_name() == "scribe_v2"


#
# Cartesia
#


@pytest.mark.asyncio
async def test_cartesia_reads_text_and_prices_from_the_reported_duration():
    model, transport = _with_transport(
        _cartesia(cost_per_minute=0.01),
        json={"text": "hello", "duration": 120.0},
    )

    text, cost = await model.a_transcribe(wav_audio())

    assert text == "hello"
    assert transport.last["path"] == "/stt"
    assert transport.last["multipart"].fields["model"] == "ink-whisper"
    assert cost == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_cartesia_reports_no_cost_without_an_explicit_rate():
    model, _ = _with_transport(_cartesia(), json={"text": "hi", "duration": 60})

    _, cost = await model.a_transcribe(wav_audio())

    assert cost is None


def test_cartesia_stt_defaults_to_a_model_the_batch_endpoint_serves():
    # ink-2 is realtime only; the batch endpoint rejects it.
    assert _cartesia().get_model_name() == "ink-whisper"


def test_cartesia_stt_pins_the_dated_api_version():
    assert _cartesia().model.headers["Cartesia-Version"] == "2026-08-14"


#
# AssemblyAI
#


@pytest.mark.asyncio
async def test_assemblyai_uses_the_one_shot_endpoint_for_a_short_turn():
    model, transport = _with_transport(_assemblyai(), json={"text": "hello"})

    text, cost = await model.a_transcribe(wav_audio(seconds=2))

    assert text == "hello"
    assert transport.last["path"] == SYNC_URL
    assert transport.last["headers"] == {"X-AAI-Model": "universal-3-5-pro"}
    assert transport.last["multipart"].file_field == "audio"
    assert cost == pytest.approx(2 / 3600 * 0.45)


@pytest.mark.asyncio
async def test_assemblyai_falls_back_to_upload_and_poll_beyond_the_sync_ceiling():
    model, transport = _with_transport(
        _assemblyai(),
        json_sequence=[
            {"upload_url": "https://cdn.assemblyai.com/upload/abc"},
            {"id": "t-1", "status": "queued"},
            {"id": "t-1", "status": "processing"},
            {"id": "t-1", "status": "completed", "text": "a long one"},
        ],
    )

    text, cost = await model.a_transcribe(wav_audio(seconds=1, duration=3600.0))

    assert text == "a long one"
    assert transport.paths() == [
        "/v2/upload",
        "/v2/transcript",
        "/v2/transcript/t-1",
        "/v2/transcript/t-1",
    ]
    # The async flow is priced per hour, well below the sync endpoint's rate.
    assert cost == pytest.approx(0.21)


@pytest.mark.asyncio
async def test_assemblyai_asks_for_the_model_as_a_fallback_list():
    model, transport = _with_transport(
        _assemblyai(),
        json_sequence=[
            {"upload_url": "https://cdn/x"},
            {"id": "t", "status": "completed", "text": "ok"},
        ],
    )

    await model.a_transcribe(wav_audio(duration=3600.0))

    # `speech_model` is deprecated in favour of the ordered list.
    body = transport.calls[1]["json"]
    assert body["speech_models"] == ["universal-3-5-pro"]
    assert "speech_model" not in body


@pytest.mark.asyncio
async def test_assemblyai_surfaces_a_failed_transcription():
    model, _ = _with_transport(
        _assemblyai(),
        json_sequence=[
            {"upload_url": "https://cdn/x"},
            {"id": "t", "status": "error", "error": "audio too quiet"},
        ],
    )

    with pytest.raises(SpeechHTTPError, match="audio too quiet"):
        await model.a_transcribe(wav_audio(duration=3600.0))


@pytest.mark.asyncio
async def test_assemblyai_takes_the_reliable_path_when_duration_is_unknown():
    # An unmeasurable clip could be longer than the sync ceiling, and the slow
    # path always works.
    model, transport = _with_transport(
        _assemblyai(),
        json_sequence=[
            {"upload_url": "https://cdn/x"},
            {"id": "t", "status": "completed", "text": "ok"},
        ],
    )
    audio = wav_audio()
    audio.dataBase64 = ""

    text, _ = await model.a_transcribe(audio)

    assert text == "ok"
    assert transport.paths()[0] == "/v2/upload"


def test_assemblyai_does_not_default_to_the_deprecated_slam_model():
    assert _assemblyai().get_model_name() == "universal-3-5-pro"


#
# The buffered partial-transcript stream, shared by every STT model
#


async def _chunks(count: int, sample_rate: int = 24000):
    # Half a second each, so two chunks cross the one-second partial threshold.
    payload = b"\x00\x00" * (sample_rate // 2)
    for index in range(count):
        yield AudioChunk.from_bytes(
            payload,
            "audio/pcm",
            sampleRate=sample_rate,
            encoding="pcm",
            final=index == count - 1,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory",
    [_deepgram, _elevenlabs, _cartesia, _assemblyai],
    ids=["deepgram", "11labs", "cartesia", "assemblyai"],
)
async def test_every_stt_model_can_produce_partial_transcripts(factory):
    # The base class raises instead of streaming, so a model that does not opt
    # in cannot be followed while the agent is still speaking.
    model = factory()
    assert model.supports_streaming() is True

    seen: List[str] = []
    calls = {"n": 0}

    async def fake_transcribe(audio, **kwargs):
        calls["n"] += 1
        return f"partial {calls['n']}", None

    model.a_transcribe = fake_transcribe
    async for text in model.a_transcribe_stream(_chunks(4)):
        seen.append(text)

    assert seen == ["partial 1", "partial 2"]


@pytest.mark.asyncio
async def test_a_partial_that_has_not_changed_is_not_re_emitted():
    model = _deepgram()

    async def fake_transcribe(audio, **kwargs):
        return "same text", None

    model.a_transcribe = fake_transcribe
    seen = [text async for text in model.a_transcribe_stream(_chunks(6))]

    assert seen == ["same text"]


@pytest.mark.asyncio
async def test_openai_still_streams_partials_through_the_shared_mixin():
    # OpenAISTTModel gave up its own copy of this loop; check it kept the
    # behaviour.
    model = OpenAISTTModel(api_key="test-key")

    async def fake_transcribe(audio, **kwargs):
        return "openai partial", None

    model.a_transcribe = fake_transcribe
    seen = [text async for text in model.a_transcribe_stream(_chunks(2))]

    assert seen == ["openai partial"]
    assert model.supports_streaming() is True


#
# Sync parity
#


@pytest.mark.parametrize(
    "factory,payload",
    [
        (_deepgram, _deepgram_payload("hi")),
        (_elevenlabs, {"text": "hi"}),
        (_cartesia, {"text": "hi"}),
        (_assemblyai, {"text": "hi"}),
    ],
    ids=["deepgram", "11labs", "cartesia", "assemblyai"],
)
def test_the_sync_path_requests_the_same_thing_as_the_async_one(
    factory, payload
):
    import asyncio

    audio = wav_audio(seconds=2)

    sync_model, sync_transport = _with_transport(factory(), json=payload)
    sync_text, _ = sync_model.transcribe(audio)

    async_model, async_transport = _with_transport(factory(), json=payload)
    async_text, _ = asyncio.run(async_model.a_transcribe(audio))

    assert sync_text == async_text
    assert jsonlib.dumps(
        sync_transport.last, sort_keys=True, default=str
    ) == jsonlib.dumps(async_transport.last, sort_keys=True, default=str)
