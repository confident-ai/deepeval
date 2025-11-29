from pydantic import SecretStr

from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.embedding_models.local_embedding_model import (
    LocalEmbeddingModel,
)
from tests.test_core.stubs import _RecordingClient


##########################
# Test Secret Management #
##########################


def test_local_embedding_model_uses_explicit_params_over_settings_and_strips_secret(
    monkeypatch,
):
    """
    Explicit ctor api_key/base_url/model must override Settings.*, and
    _build_client should receive a plain string api_key even if Settings
    stores a SecretStr.
    """
    # Seed env so Settings sees baseline values
    monkeypatch.setenv("LOCAL_EMBEDDING_API_KEY", "env-secret-key")
    monkeypatch.setenv(
        "LOCAL_EMBEDDING_BASE_URL", "http://settings-host:11434/v1"
    )
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL_NAME", "settings-embedding-model")

    # Rebuild Settings from env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity: API key should be a SecretStr on the settings object
    assert isinstance(settings.LOCAL_EMBEDDING_API_KEY, SecretStr)

    # Explicit ctor args should override everything from Settings
    model = LocalEmbeddingModel(
        api_key="ctor-secret-key",
        base_url="http://ctor-host:11434/v1",
        model="ctor-embedding-model",
    )

    # Directly exercise _build_client with our recording stub
    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    # Client sees ctor api_key as a plain string
    api_key = kw.get("api_key")
    assert isinstance(api_key, str)
    assert api_key == "ctor-secret-key"

    # Base URL should come from ctor as well
    base_url = kw.get("base_url")
    assert base_url is not None
    assert base_url.rstrip("/") == "http://ctor-host:11434/v1"

    # Model name should match the ctor-provided model
    assert model.model_name == "ctor-embedding-model"


def test_local_embedding_model_defaults_from_settings(monkeypatch):
    """
    When no ctor args are provided, LocalEmbeddingModel should pull its
    configuration (API key, base_url, model_name) from Settings, which
    in turn are backed by env vars.
    """
    # Seed env so Settings picks up all Local-embedding-related values
    monkeypatch.setenv("LOCAL_EMBEDDING_API_KEY", "env-secret-key")
    monkeypatch.setenv(
        "LOCAL_EMBEDDING_BASE_URL", "http://settings-host:11434/v1"
    )
    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL_NAME", "settings-embedding-model")

    # Rebuild settings from env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity: API key should be a SecretStr on the settings object
    assert isinstance(settings.LOCAL_EMBEDDING_API_KEY, SecretStr)

    # No ctor args: everything should come from Settings
    model = LocalEmbeddingModel()

    # Directly exercise _build_client to verify the resolved kwargs
    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    # API key is unwrapped to a plain string from Settings
    api_key = kw.get("api_key")
    assert isinstance(api_key, str)
    assert api_key == "env-secret-key"

    # Base URL from Settings (allow trailing slash differences)
    base_url = kw.get("base_url")
    assert base_url is not None
    assert base_url.rstrip("/") == "http://settings-host:11434/v1"

    # Model name should also come from Settings
    assert model.model_name == "settings-embedding-model"
