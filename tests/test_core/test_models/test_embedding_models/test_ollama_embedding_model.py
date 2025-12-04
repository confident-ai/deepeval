from unittest.mock import patch

from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.embedding_models.ollama_embedding_model import (
    OllamaEmbeddingModel,
)
from tests.test_core.stubs import _RecordingClient, make_fake_ollama_module


@patch(
    "deepeval.models.embedding_models.ollama_embedding_model.require_dependency"
)
def test_ollama_embedding_model_uses_explicit_params_over_settings(
    mock_require_dep, settings
):
    """
    Explicit ctor host/model must override Settings.*, and the underlying
    Ollama client must be constructed with the ctor host even if Settings
    provides defaults.
    """
    # Seed env so Settings sees baseline values
    with settings.edit(persist=False):
        settings.LOCAL_EMBEDDING_BASE_URL = "http://settings-host:11434"
        settings.LOCAL_EMBEDDING_MODEL_NAME = "settings-embedding-model"

    # Rebuild Settings from env
    reset_settings(reload_dotenv=False)
    _ = get_settings()

    # Fake ollama module returned by require_dependency
    fake_ollama = make_fake_ollama_module(_RecordingClient)
    mock_require_dep.return_value = fake_ollama

    # Explicit ctor args should override everything from Settings
    model = OllamaEmbeddingModel(
        model="ctor-embedding-model",
        base_url="http://ctor-host:11434",
    )

    # Exercise load_model() so we go through require_dependency + _build_client
    client = model.load_model()
    kw = client.kwargs

    # Host should come from ctor, not Settings
    host = kw.get("host")
    assert host is not None
    assert host.rstrip("/") == "http://ctor-host:11434"

    # Model name should be the ctor-provided value
    assert model.name == "ctor-embedding-model"

    # ensure we actually called require_dependency
    mock_require_dep.assert_any_call(
        "ollama",
        provider_label="OllamaEmbeddingModel",
        install_hint="Install it with `pip install ollama`.",
    )


@patch(
    "deepeval.models.embedding_models.ollama_embedding_model.require_dependency"
)
def test_ollama_embedding_model_defaults_from_settings(
    mock_require_dep,
    settings,
):
    """
    When no ctor args are provided, OllamaEmbeddingModel should pull host
    and model from Settings, which are backed by env vars.
    """
    # Seed env so Settings picks up Ollama-related defaults
    with settings.edit(persist=False):
        settings.LOCAL_EMBEDDING_BASE_URL = "http://settings-host:11434"
        settings.LOCAL_EMBEDDING_MODEL_NAME = "settings-embedding-model"

    # Rebuild Settings from env
    reset_settings(reload_dotenv=False)
    _ = get_settings()

    # Fake ollama module returned by require_dependency
    fake_ollama = make_fake_ollama_module(_RecordingClient)
    mock_require_dep.return_value = fake_ollama

    # No ctor args: everything should come from Settings
    model = OllamaEmbeddingModel()

    # Exercise load_model() so we go through require_dependency + _build_client
    client = model.load_model()
    kw = client.kwargs

    # Host comes from Settings (allow for trailing slash differences)
    host = kw.get("host")
    assert host is not None
    assert host.rstrip("/") == "http://settings-host:11434"

    # Model name should also come from Settings
    assert model.name == "settings-embedding-model"

    mock_require_dep.assert_any_call(
        "ollama",
        provider_label="OllamaEmbeddingModel",
        install_hint="Install it with `pip install ollama`.",
    )


########################################################
# Test legacy keyword backwards compatability behavior #
########################################################


def test_ollama_embedding_model_accepts_legacy_host_keyword_and_maps_to_base_url():
    """
    Using the legacy `model` keyword should still work:
    - It should populate `model`
    - It should not be forwarded through `model.kwargs`
    """
    model = OllamaEmbeddingModel(host="ctor-host")

    # legacy keyword mapped to canonical parameter
    assert model.base_url == "ctor-host"

    # legacy key should not be forwarded to the client kwargs
    assert "host" not in model.kwargs
