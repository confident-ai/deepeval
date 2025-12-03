from unittest.mock import patch
from pydantic import SecretStr

from deepeval.config.settings import reset_settings
from deepeval.models.mlllms.openai_model import (
    MultimodalOpenAIModel,
    default_multimodal_gpt_model,
)


##########################
# Test Secret Management #
##########################


@patch("deepeval.models.mlllms.openai_model.OpenAI")
def test_multimodal_openai_model_uses_explicit_key_over_settings_and_strips_secret(
    mock_openai_cls, settings
):
    """
    Explicit ctor `_openai_api_key` must override Settings.OPENAI_API_KEY, and
    the client should see a plain string even though Settings stores a
    SecretStr.
    """
    # Seed env so Settings sees an OPENAI_API_KEY
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "env-secret-key"

    # settings should expose this as a SecretStr
    assert isinstance(settings.OPENAI_API_KEY, SecretStr)

    # Construct the model with an explicit key
    MultimodalOpenAIModel(
        model_name="gpt-4o",
        api_key="ctor-secret-key",
    )

    # Ensure OpenAI client was called once
    mock_openai_cls.assert_called_once()

    # Check the arguments passed to OpenAI client
    kwargs = mock_openai_cls.call_args.kwargs
    api_key = kwargs.get("api_key")

    # Client must see the ctor key, as a plain string
    assert isinstance(api_key, str)
    assert api_key == "ctor-secret-key"


@patch("deepeval.models.mlllms.openai_model.OpenAI")
def test_multimodal_openai_model_defaults_key_from_settings(
    mock_openai_cls, settings
):
    """
    When no ctor `_openai_api_key` is provided, MultimodalOpenAIModel should
    pull the API key from Settings.OPENAI_API_KEY (backed by env) and unwrap
    it to a plain string for the client.
    """
    # Seed env so Settings picks up OPENAI_API_KEY
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "env-secret-key"

    # settings should expose this as a SecretStr
    assert isinstance(settings.OPENAI_API_KEY, SecretStr)

    MultimodalOpenAIModel(model_name="gpt-4o")

    # Ensure OpenAI client was called once
    mock_openai_cls.assert_called_once()

    # Check the arguments passed to OpenAI client
    kwargs = mock_openai_cls.call_args.kwargs
    api_key = kwargs.get("api_key")

    # Client must see the ctor key, as a plain string
    assert isinstance(api_key, str)
    assert api_key == "env-secret-key"


################################
# Test model param / fallback  #
################################


def test_multimodal_openai_model_uses_explicit_model_over_settings(settings):
    """
    Explicit ctor `model` must override Settings.OPENAI_MODEL_NAME.
    """
    # Seed env for both API key + MLLM model
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "env-secret-key"
        settings.OPENAI_MODEL_NAME = "gpt-4o"

    # settings contains our seeded model name
    assert settings.OPENAI_MODEL_NAME == "gpt-4o"

    # Explicit model should win over Settings.OPENAI_MODEL_NAME
    model = MultimodalOpenAIModel(model_name="gpt-4.1")

    # Parsed/validated model name should match ctor value
    assert model.model_name == "gpt-4.1"


def test_multimodal_openai_model_defaults_model_from_settings(settings):
    """
    When ctor `model` is None, MultimodalOpenAIModel should pull the model
    name from Settings.OPENAI_MODEL_NAME.
    """
    # Seed env so Settings picks up the MLLM model name and API key
    with settings.edit(persist=False):
        settings.OPENAI_API_KEY = "env-secret-key"
        settings.OPENAI_MODEL_NAME = "gpt-4o"

    # settings contains our seeded model name
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


########################################################
# Test legacy keyword backwards compatability behavior #
########################################################


def test_multimodal_openai_model_accepts_legacy_model_keyword_and_maps_to_model_name():
    """
    Using the legacy `model` keyword should still work:
    - It should populate `model_name`
    - It should not be forwarded through `model.kwargs`
    """

    model = MultimodalOpenAIModel(model="gpt-4o")

    # legacy keyword mapped to canonical parameter
    assert model.model_name == "gpt-4o"

    # legacy key should not be forwarded to the client kwargs
    assert "model" not in model.kwargs


def test_multimodal_openai_model_accepts_legacy__openai_api_key_keyword_and_maps_to_api_key():
    """
    Using the legacy `model` keyword should still work:
    - It should populate `model_name`
    - It should not be forwarded through `model.kwargs`
    """

    model = MultimodalOpenAIModel(_openai_api_key="test-key")

    # legacy keyword mapped to canonical parameter
    assert model.api_key and model.api_key.get_secret_value() == "test-key"

    # legacy key should not be forwarded to the client kwargs
    assert "api_key" not in model.kwargs
