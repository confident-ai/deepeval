"""The per-provider schema models and the two helpers that cross the wire."""

import warnings

import pytest

from deepeval.models import (
    CartesiaSTTModel,
    CartesiaTTSModel,
    DeepgramSTTModel,
    DeepgramTTSModel,
    ElevenLabsSTTModel,
    ElevenLabsTTSModel,
)
from deepeval.models.speech import (
    SpeechHTTPError,
    dump_request,
    parse_response,
)
from deepeval.models.stt.assemblyai.schema import (
    AssemblyAITranscriptRequest,
    AssemblyAITranscriptResponse,
    AssemblyAIUploadResponse,
)
from deepeval.models.stt.cartesia.schema import CartesiaSTTResponse
from deepeval.models.stt.deepgram.schema import (
    DeepgramListenParams,
    DeepgramListenResponse,
)
from deepeval.models.stt.elevenlabs.schema import ElevenLabsSTTResponse
from deepeval.models.tts.cartesia.schema import (
    CartesiaOutputFormat,
    CartesiaTTSRequest,
)
from deepeval.models.tts.elevenlabs.schema import ElevenLabsTTSRequest
from tests.test_core.test_models.speech_stubs import FakeTransport, wav_audio


#
# dump_request
#


def test_unset_optional_fields_are_left_off_the_wire():
    # Several of these APIs read an explicit null as an error, so a schema can
    # only declare every optional parameter if unset ones are dropped.
    dumped = dump_request(
        ElevenLabsTTSRequest(text="hi", model_id="eleven_flash_v2_5")
    )

    assert dumped == {"text": "hi", "model_id": "eleven_flash_v2_5"}


def test_a_set_optional_field_is_kept():
    dumped = dump_request(
        ElevenLabsTTSRequest(
            text="hi", model_id="m", language_code="es", voice_settings={"s": 1}
        )
    )

    assert dumped["language_code"] == "es"
    assert dumped["voice_settings"] == {"s": 1}


def test_unknown_fields_survive_so_new_provider_params_are_reachable():
    # This is what makes generation_kwargs useful before deepeval has a field
    # for a newly shipped provider parameter.
    dumped = dump_request(
        ElevenLabsTTSRequest(
            text="hi", model_id="m", some_brand_new_flag=True, seed=7
        )
    )

    assert dumped["some_brand_new_flag"] is True
    assert dumped["seed"] == 7


def test_nested_objects_are_serialized_as_objects():
    dumped = dump_request(
        CartesiaTTSRequest(
            model_id="sonic-3.6",
            transcript="hi",
            voice="v",
            output_format=CartesiaOutputFormat(
                container="raw", sample_rate=24000
            ),
        )
    )

    assert dumped["output_format"] == {
        "container": "raw",
        "sample_rate": 24000,
        "encoding": "pcm_s16le",
    }


def test_a_declared_field_given_the_wrong_type_is_rejected():
    # Modelling requests is what buys this: a typo in generation_kwargs that
    # collides with a real field fails here rather than at the provider.
    with pytest.raises(Exception, match="sample_rate"):
        CartesiaOutputFormat(container="raw", sample_rate="not-a-number")


def test_model_prefixed_fields_do_not_warn_about_pydantics_namespace():
    # ElevenLabs and Cartesia both name a field `model_id`, which collides with
    # pydantic's reserved `model_` prefix unless the schema clears it.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ElevenLabsTTSRequest(text="hi", model_id="m")
        CartesiaTTSRequest(
            model_id="m",
            transcript="hi",
            voice="v",
            output_format=CartesiaOutputFormat(
                container="raw", sample_rate=8000
            ),
        )
        DeepgramListenParams(model="nova-3")


#
# parse_response
#


def test_unknown_response_fields_are_tolerated():
    # Providers add fields without warning; a run should not care.
    payload = parse_response(
        CartesiaSTTResponse,
        {"text": "hi", "duration": 1.0, "a_field_from_next_year": {"x": 1}},
        provider_label="Cartesia",
    )

    assert payload.text == "hi"


def test_a_response_of_the_wrong_shape_names_the_provider_and_the_cause():
    with pytest.raises(SpeechHTTPError) as excinfo:
        parse_response(
            CartesiaSTTResponse,
            {"duration": "not-a-number"},
            provider_label="Cartesia",
        )

    message = str(excinfo.value)
    assert "Cartesia" in message
    assert "changed its response format" in message
    assert "duration" in message


def test_a_missing_required_response_field_is_reported():
    with pytest.raises(SpeechHTTPError, match="AssemblyAI"):
        parse_response(
            AssemblyAIUploadResponse, {}, provider_label="AssemblyAI"
        )


#
# Silence vs. a changed response shape
#


@pytest.mark.parametrize(
    "schema,silent,absent",
    [
        (CartesiaSTTResponse, {"text": ""}, {"duration": 1.0}),
        (ElevenLabsSTTResponse, {"text": ""}, {"audio_duration_secs": 1.0}),
        (AssemblyAITranscriptResponse, {"text": ""}, {"status": "completed"}),
        (
            DeepgramListenResponse,
            {"results": {"channels": [{"alternatives": [{"transcript": ""}]}]}},
            {"results": {"channels": []}},
        ),
    ],
    ids=["cartesia", "11labs", "assemblyai", "deepgram"],
)
def test_transcript_tells_silence_apart_from_a_missing_field(
    schema, silent, absent
):
    # Silence is a legitimate answer and comes back as an empty string. `None`
    # means the field deepeval reads was not in the response at all, which is
    # what a changed response format would look like.
    assert schema.model_validate(silent).transcript() == ""
    assert schema.model_validate(absent).transcript() is None


def test_elevenlabs_falls_back_to_per_channel_transcripts():
    payload = ElevenLabsSTTResponse.model_validate(
        {"transcripts": [{"text": "left"}, {"text": "right"}]}
    )

    assert payload.transcript() == "left right"


def test_deepgram_reads_the_first_alternative_it_finds():
    payload = DeepgramListenResponse.model_validate(
        {
            "results": {
                "channels": [
                    {"alternatives": []},
                    {"alternatives": [{"transcript": "second channel"}]},
                ]
            }
        }
    )

    assert payload.transcript() == "second channel"


#
# AssemblyAI poll-loop helpers
#


@pytest.mark.parametrize(
    "status,finished,failed",
    [
        ("queued", False, False),
        ("processing", False, False),
        ("completed", True, False),
        ("error", True, True),
        # The synchronous endpoint returns a finished transcript with no queue
        # to report on, so it sends no status at all.
        (None, True, False),
    ],
)
def test_status_drives_the_poll_loop(status, finished, failed):
    payload = AssemblyAITranscriptResponse.model_validate(
        {"status": status} if status else {}
    )

    assert payload.finished is finished
    assert payload.failed is failed


def test_assemblyai_asks_for_models_as_an_ordered_list():
    # The singular `speech_model` field is deprecated.
    dumped = dump_request(
        AssemblyAITranscriptRequest(
            audio_url="https://cdn/x", speech_models=["universal-3-5-pro"]
        )
    )

    assert dumped == {
        "audio_url": "https://cdn/x",
        "speech_models": ["universal-3-5-pro"],
    }


#
# The models actually route their kwargs through the schemas
#


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory,kwarg,where",
    [
        (
            lambda **kw: ElevenLabsTTSModel(api_key="k", **kw),
            "generation_kwargs",
            "json",
        ),
        (
            lambda **kw: DeepgramTTSModel(api_key="k", **kw),
            "generation_kwargs",
            "params",
        ),
        (
            lambda **kw: CartesiaTTSModel(api_key="k", voice="v", **kw),
            "generation_kwargs",
            "json",
        ),
    ],
    ids=["11labs", "deepgram", "cartesia"],
)
async def test_generation_kwargs_reach_the_wire_through_the_schema(
    factory, kwarg, where
):
    model = factory(**{kwarg: {"an_unmodelled_provider_flag": True}})
    transport = FakeTransport(content=b"RIFF")
    model.model = transport

    await model.a_synthesize("hi")

    assert transport.last[where]["an_unmodelled_provider_flag"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "factory,where",
    [
        (lambda **kw: DeepgramSTTModel(api_key="k", **kw), "params"),
        (lambda **kw: ElevenLabsSTTModel(api_key="k", **kw), "multipart"),
        (lambda **kw: CartesiaSTTModel(api_key="k", **kw), "multipart"),
    ],
    ids=["deepgram", "11labs", "cartesia"],
)
async def test_transcription_kwargs_reach_the_wire_through_the_schema(
    factory, where
):
    model = factory(transcription_kwargs={"an_unmodelled_provider_flag": True})
    transport = FakeTransport(
        json={
            "text": "hi",
            "results": {"channels": [{"alternatives": [{"transcript": "hi"}]}]},
        }
    )
    model.model = transport

    await model.a_transcribe(wav_audio())

    sent = transport.last[where]
    fields = sent.fields if where == "multipart" else sent
    assert fields["an_unmodelled_provider_flag"] is True
