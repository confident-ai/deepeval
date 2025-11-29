from pydantic import SecretStr

from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.mlllms.openai_model import (
    MultimodalOpenAIModel,
    default_multimodal_gpt_model,
)
from tests.test_core.stubs import _RecordingClient


##########################
# Test Secret Management #
##########################


def test_multimodal_openai_model_uses_explicit_key_over_settings_and_strips_secret(
    monkeypatch,
):
    """
    Explicit ctor `_openai_api_key` must override Settings.OPENAI_API_KEY, and
    _build_client should see a plain string even though Settings stores a
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
    model = MultimodalOpenAIModel(
        model="gpt-4o",
        _openai_api_key="ctor-secret-key",
    )

    # Directly exercise _build_client with our recording stub
    client = model._build_client(_RecordingClient)
    api_key = client.kwargs.get("api_key")

    # Client must see the ctor key, as a plain string
    assert isinstance(api_key, str)
    assert api_key == "ctor-secret-key"


def test_multimodal_openai_model_defaults_key_from_settings(monkeypatch):
    """
    When no ctor `_openai_api_key` is provided, MultimodalOpenAIModel should
    pull the API key from Settings.OPENAI_API_KEY (backed by env) and unwrap
    it to a plain string for the client.
    """
    # Seed env so Settings picks up OPENAI_API_KEY
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret-key")

    # Rebuild settings from env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity: Settings should expose this as a SecretStr
    assert isinstance(settings.OPENAI_API_KEY, SecretStr)

    # No ctor key: everything should come from Settings
    model = MultimodalOpenAIModel(model="gpt-4o")

    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    # Client kwargs pulled from Settings, unwrapped to str
    api_key = kw.get("api_key")
    assert isinstance(api_key, str)
    assert api_key == "env-secret-key"


################################
# Test model param / fallback  #
################################


def test_multimodal_openai_model_uses_explicit_model_over_settings(monkeypatch):
    """
    Explicit ctor `model` must override Settings.OPENAI_MODEL_NAME.
    """
    # Seed env for both API key + MLLM model
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret-key")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4o")

    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity: settings contains our seeded model name
    assert settings.OPENAI_MODEL_NAME == "gpt-4o"

    # Explicit model should win over Settings.OPENAI_MODEL_NAME
    model = MultimodalOpenAIModel(model="gpt-4.1")

    # Parsed/validated model name should match ctor value
    assert model.model_name == "gpt-4.1"


def test_multimodal_openai_model_defaults_model_from_settings(monkeypatch):
    """
    When ctor `model` is None, MultimodalOpenAIModel should pull the model
    name from Settings.OPENAI_MODEL_NAME.
    """
    # Seed env so Settings picks up the MLLM model name and API key
    monkeypatch.setenv("OPENAI_API_KEY", "env-secret-key")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-4o")

    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity: settings contains our seeded model name
    assert settings.OPENAI_MODEL_NAME == "gpt-4o"

    # No ctor model: model name should come from settings
    model = MultimodalOpenAIModel()

    assert model.model_name == "gpt-4o"


def test_multimodal_openai_model_uses_default_when_no_model_config(monkeypatch):
    """
    If both ctor `model` and Settings.OPENAI_MODEL_NAME are None,
    MultimodalOpenAIModel should use default_multimodal_gpt_model.
    """
    # Ensure no model name is available from env-backed settings
    monkeypatch.delenv("OPENAI_MODEL_NAME", raising=False)
    # API key can be absent for this test
    reset_settings(reload_dotenv=False)

    model = MultimodalOpenAIModel()
    assert model.model_name == default_multimodal_gpt_model
