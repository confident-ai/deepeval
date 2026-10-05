"""Which speech provider and model a run ends up with.

TTS and STT are picked the same way LLM providers are: a `USE_*` flag chooses
the provider, `DEEPEVAL_TTS_MODEL` / `DEEPEVAL_STT_MODEL` choose the model
name, and anything passed in code outranks both. These pin down that order,
and that the two families never reach into each other.
"""

import pytest

from deepeval.models.speech_selection import (
    initialize_stt_model,
    initialize_tts_model,
)
from deepeval.models.stt import (
    AssemblyAISTTModel,
    DeepgramSTTModel,
    OpenAISTTModel,
)
from deepeval.models.tts import (
    DeepgramTTSModel,
    ElevenLabsTTSModel,
    OpenAITTSModel,
)


@pytest.fixture(autouse=True)
def speech_api_keys(monkeypatch):
    # Every provider raises on construction without a key, and none of these
    # tests make a request.
    for name in (
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
        "CARTESIA_API_KEY",
        "DEEPGRAM_API_KEY",
        "ASSEMBLYAI_API_KEY",
    ):
        monkeypatch.setenv(name, "test-key")


def test_no_flag_falls_back_to_openai():
    assert isinstance(initialize_tts_model(), OpenAITTSModel)
    assert isinstance(initialize_stt_model(), OpenAISTTModel)


def test_flag_picks_the_provider(monkeypatch):
    monkeypatch.setenv("USE_DEEPGRAM_TTS", "1")
    monkeypatch.setenv("USE_ASSEMBLYAI_STT", "1")

    assert isinstance(initialize_tts_model(), DeepgramTTSModel)
    assert isinstance(initialize_stt_model(), AssemblyAISTTModel)


def test_family_env_var_picks_the_model(monkeypatch):
    monkeypatch.setenv("USE_DEEPGRAM_STT", "1")
    monkeypatch.setenv("DEEPEVAL_STT_MODEL", "nova-3-general")

    model = initialize_stt_model()

    assert isinstance(model, DeepgramSTTModel)
    assert model.get_model_name() == "nova-3-general"


def test_a_string_names_the_model_for_the_selected_provider(monkeypatch):
    monkeypatch.setenv("USE_ELEVENLABS_TTS", "1")
    monkeypatch.setenv("DEEPEVAL_TTS_MODEL", "eleven_flash_v2_5")

    model = initialize_tts_model("eleven_turbo_v2_5")

    # The string is the model name only: the provider still comes from the
    # flag, and the name beats the env var.
    assert isinstance(model, ElevenLabsTTSModel)
    assert model.get_model_name() == "eleven_turbo_v2_5"


def test_an_instance_passes_through_untouched(monkeypatch):
    monkeypatch.setenv("USE_DEEPGRAM_TTS", "1")
    passed = OpenAITTSModel(model="tts-1-hd")

    assert initialize_tts_model(passed) is passed


def test_the_two_families_are_independent(monkeypatch):
    monkeypatch.setenv("USE_DEEPGRAM_TTS", "1")
    monkeypatch.setenv("DEEPEVAL_TTS_MODEL", "aura-2-thalia-en")

    # No STT flag, so STT stays on OpenAI and ignores the TTS model name.
    stt = initialize_stt_model()

    assert isinstance(initialize_tts_model(), DeepgramTTSModel)
    assert isinstance(stt, OpenAISTTModel)
    assert stt.get_model_name() != "aura-2-thalia-en"


def test_voice_config_resolves_through_the_same_chain(monkeypatch):
    monkeypatch.setenv("USE_DEEPGRAM_STT", "1")
    from deepeval.voice import VoiceConfig
    from deepeval.voice.connectors.transports.callback import (
        CallbackVoiceConnector,
    )

    async def agent(audio):
        raise AssertionError("the agent is never called in this test")

    config = VoiceConfig(connector=CallbackVoiceConnector(agent))

    assert isinstance(config.stt_model, DeepgramSTTModel)
    assert isinstance(config.tts_model, OpenAITTSModel)
