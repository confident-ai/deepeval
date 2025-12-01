from pydantic import SecretStr

from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.embedding_models.openai_embedding_model import (
    OpenAIEmbeddingModel,
)
from tests.test_core.stubs import _RecordingClient


##########################
# Test Secret Management #
##########################


def test_openai_embedding_model_uses_explicit_key_over_settings_and_strips_secret(
    monkeypatch,
):
    """
    Explicit ctor openai_api_key must override Settings.OPENAI_API_KEY, and
    _build_client should see a plain string, even though Settings stores a
    SecretStr.
    """
    # Seed env so Settings sees an OPENAI_API_KEY
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret-key")

    # Rebuild the Settings singleton from the current env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity check: Settings should expose this as a SecretStr
    assert isinstance(settings.OPENAI_API_KEY, SecretStr)

    # Construct the model with an explicit key
    model = OpenAIEmbeddingModel(
        model="text-embedding-3-small",
        openai_api_key="ctor-secret-key",
    )

    # Directly exercise _build_client with our recording stub
    client = model._build_client(_RecordingClient)
    api_key = client.kwargs.get("api_key")

    # Client must see the ctor key, as a plain string
    assert isinstance(api_key, str)
    assert api_key == "ctor-secret-key"


def test_openai_embedding_model_defaults_from_settings(monkeypatch):
    """
    When no ctor openai_api_key is provided, OpenAIEmbeddingModel should pull
    the API key from Settings.OPENAI_API_KEY (backed by env).
    """
    # Seed env so Settings picks up OPENAI_API_KEY
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret-key")

    # Rebuild settings from env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity: Settings should expose this as a SecretStr
    assert isinstance(settings.OPENAI_API_KEY, SecretStr)

    # No ctor api_key: everything should come from Settings
    model = OpenAIEmbeddingModel(model="text-embedding-3-small")

    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    # Client kwargs pulled from Settings
    api_key = kw.get("api_key")
    assert isinstance(api_key, str)
    assert api_key == "env-secret-key"
