from pydantic import SecretStr

from deepeval.config.settings import reset_settings
from deepeval.models.mlllms.openai_model import (
    MultimodalOpenAIModel,
    default_multimodal_gpt_model,
)
from tests.test_core.stubs import _RecordingClient


##########################
# Test Secret Management #
##########################


def test_multimodal_openai_model_uses_explicit_key_over_settings_and_strips_secret(
    settings,
):
    """
    Explicit ctor `_openai_api_key` must override Settings.OPENAI_API_KEY, and
    _build_client should see a plain string even though Settings stores a
    SecretStr.
    """
    # Seed settings with OPENAI_API_KEY
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "env-secret-key"

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


def test_multimodal_openai_model_defaults_key_from_settings(settings):
    """
    When no ctor `_openai_api_key` is provided, MultimodalOpenAIModel should
    pull the API key from Settings.OPENAI_API_KEY (backed by env) and unwrap
    it to a plain string for the client.
    """
    # Seed settings with OPENAI_API_KEY
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "env-secret-key"

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


def test_multimodal_openai_model_uses_explicit_model_over_settings(settings):
    """
    Explicit ctor `model` must override Settings.OPENAI_MODEL_NAME.
    """
    # Seed settings
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "env-secret-key"
        settings.OPENAI_MODEL_NAME = "gpt-4o"

    # Sanity: settings contains our seeded model name
    assert settings.OPENAI_MODEL_NAME == "gpt-4o"

    # Explicit model should win over Settings.OPENAI_MODEL_NAME
    model = MultimodalOpenAIModel(model="gpt-4.1")

    # Parsed/validated model name should match ctor value
    assert model.model_name == "gpt-4.1"


def test_multimodal_openai_model_defaults_model_from_settings(settings):
    """
    When ctor `model` is None, MultimodalOpenAIModel should pull the model
    name from Settings.OPENAI_MODEL_NAME.
    """
    # Seed settings
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "env-secret-key"
        settings.OPENAI_MODEL_NAME = "gpt-4o"

    # Sanity: settings contains our seeded model name
    assert settings.OPENAI_MODEL_NAME == "gpt-4o"

    # No ctor model: model name should come from settings
    model = MultimodalOpenAIModel()

    assert model.model_name == "gpt-4o"


def test_multimodal_openai_model_uses_default_when_no_model_config(
    monkeypatch, settings
):
    """
    If both ctor `model` and Settings.OPENAI_MODEL_NAME are None,
    MultimodalOpenAIModel should use default_multimodal_gpt_model.
    """

    # Seed settings
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "env-secret-key"

    # Ensure no model name is available from env-backed settings
    monkeypatch.delenv("OPENAI_MODEL_NAME", raising=False)
    # API key can be absent for this test
    reset_settings(reload_dotenv=False)

    model = MultimodalOpenAIModel()
    assert model.model_name == default_multimodal_gpt_model
