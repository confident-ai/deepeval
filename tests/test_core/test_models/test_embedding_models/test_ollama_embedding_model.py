"""Tests for OllamaEmbeddingModel settings/host + model handling."""

from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.embedding_models.ollama_embedding_model import (
    OllamaEmbeddingModel,
)
from tests.test_core.stubs import _RecordingClient


def test_ollama_embedding_model_uses_explicit_params_over_settings(monkeypatch):
    """
    Explicit ctor host/model must override Settings.*, and _build_client
    should receive the ctor host even if Settings provides defaults.
    """
    # Seed env so Settings sees baseline values
    monkeypatch.setenv("LOCAL_EMBEDDING_BASE_URL", "http://settings-host:11434")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL_NAME", "settings-embedding-model")

    # Rebuild Settings from env (not strictly required for these tests,
    # but keeps behavior consistent with other embedding tests)
    reset_settings(reload_dotenv=False)
    _ = get_settings()

    # Explicit ctor args should override everything from Settings
    model = OllamaEmbeddingModel(
        model="ctor-embedding-model",
        base_url="http://ctor-host:11434",
    )

    # Directly exercise _build_client to verify resolved kwargs
    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    # Host should come from ctor, not Settings
    host = kw.get("host")
    assert host is not None
    assert host.rstrip("/") == "http://ctor-host:11434"

    # Model name should be the ctor-provided value
    assert model.name == "ctor-embedding-model"


def test_ollama_embedding_model_defaults_from_settings(monkeypatch):
    """
    When no ctor args are provided, OllamaEmbeddingModel should pull host
    and model from Settings, which are backed by env vars.
    """
    # Seed env so Settings picks up Ollama-related defaults
    monkeypatch.setenv("LOCAL_EMBEDDING_BASE_URL", "http://settings-host:11434")
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL_NAME", "settings-embedding-model")

    # Rebuild Settings from env
    reset_settings(reload_dotenv=False)
    _ = get_settings()

    # No ctor args: everything should come from Settings
    model = OllamaEmbeddingModel()

    # Directly exercise _build_client to verify resolved kwargs
    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    # Host comes from Settings (allow for trailing slash differences)
    host = kw.get("host")
    assert host is not None
    assert host.rstrip("/") == "http://settings-host:11434"

    # Model name should also come from Settings
    assert model.name == "settings-embedding-model"


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
