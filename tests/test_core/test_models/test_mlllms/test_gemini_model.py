from unittest.mock import patch
from pydantic import SecretStr

from deepeval.models.mlllms.gemini_model import MultimodalGeminiModel
from tests.test_core.stubs import _make_fake_genai_module


@patch("deepeval.models.mlllms.gemini_model.require_dependency")
def test_multimodal_gemini_direct_uses_explicit_api_key_over_settings(
    mock_require_dep, settings
):
    """
    Explicit ctor `api_key` must override Settings.GOOGLE_API_KEY when using the
    direct Gemini API path (non-Vertex), and `_build_client` should pass the ctor
    key through to genai.Client.
    """
    with settings.edit(persist=False):
        settings.GEMINI_MODEL_NAME = "settings-gemini-model"
        settings.GOOGLE_API_KEY = "env-google-api-key"

    # When MultimodalGeminiModel calls require_dependency, return our fake module
    mock_require_dep.return_value = _make_fake_genai_module()

    # Settings should expose this as a SecretStr
    assert isinstance(settings.GOOGLE_API_KEY, SecretStr)

    # Explicit ctor args should win over Settings.*
    model = MultimodalGeminiModel(
        model_name="ctor-gemini-model",
        api_key="ctor-google-api-key",
    )

    client = model.model
    kwargs = client.kwargs

    # Client must see the ctor api_key, not the one from Settings
    assert kwargs.get("api_key") == "ctor-google-api-key"
    # Direct path should not be constructed as Vertex AI
    assert kwargs.get("vertexai") is None or kwargs.get("vertexai") is False

    # Model name should reflect the ctor-provided value
    assert model.model_name == "ctor-gemini-model"


@patch("deepeval.models.mlllms.gemini_model.require_dependency")
def test_multimodal_gemini_direct_defaults_api_key_from_settings(
    mock_require_dep, settings
):
    """
    When no ctor `api_key` is provided, MultimodalGeminiModel should pull the
    API key and model name from Settings (backed by env), and `_build_client`
    should pass that key to genai.Client.
    """
    with settings.edit(persist=False):
        settings.GEMINI_MODEL_NAME = "settings-gemini-model"
        settings.GOOGLE_API_KEY = "env-google-api-key"

    # When MultimodalGeminiModel calls require_dependency, return our fake module
    mock_require_dep.return_value = _make_fake_genai_module()

    model = MultimodalGeminiModel()

    client = model.model
    kwargs = client.kwargs

    # API key comes from Settings
    assert kwargs.get("api_key") == "env-google-api-key"

    # Model name should default from Settings
    assert model.model_name == "settings-gemini-model"


@patch("deepeval.models.mlllms.gemini_model.require_dependency")
def test_multimodal_gemini_vertexai_uses_explicit_project_and_location_over_settings(
    mock_require_dep, settings
):
    """
    When Vertex AI is used, explicit ctor `project` and `location` must override
    Settings.GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION, and `_build_client`
    should pass those through to genai.Client.
    """
    with settings.edit(persist=False):
        settings.GEMINI_MODEL_NAME = "settings-gemini-model"
        settings.GOOGLE_CLOUD_PROJECT = "settings-project"
        settings.GOOGLE_CLOUD_LOCATION = "settings-location"

    # When MultimodalGeminiModel calls require_dependency, return our fake module
    mock_require_dep.return_value = _make_fake_genai_module()

    model = MultimodalGeminiModel(
        project="ctor-project",
        location="ctor-location",
    )

    client = model.model
    kwargs = client.kwargs

    # Vertex AI mode should be enabled
    assert kwargs.get("vertexai") is True

    # Project/location must come from ctor args, not Settings
    assert kwargs.get("project") == "ctor-project"
    assert kwargs.get("location") == "ctor-location"


@patch("deepeval.models.mlllms.gemini_model.require_dependency")
def test_multimodal_gemini_vertexai_defaults_project_and_location_from_settings(
    mock_require_dep, settings
):
    """
    When Vertex AI is used and no ctor project/location are provided,
    MultimodalGeminiModel should pull project and location from Settings,
    and `_build_client` should pass those through to genai.Client.
    """
    with settings.edit(persist=False):
        settings.GEMINI_MODEL_NAME = "settings-gemini-model"
        settings.GOOGLE_CLOUD_PROJECT = "settings-project"
        settings.GOOGLE_CLOUD_LOCATION = "settings-location"

    # When MultimodalGeminiModel calls require_dependency, return our fake module
    mock_require_dep.return_value = _make_fake_genai_module()

    model = MultimodalGeminiModel()

    client = model.model
    kwargs = client.kwargs

    # Vertex AI mode should be enabled
    assert kwargs.get("vertexai") is True

    # Project/location should come from Settings
    assert kwargs.get("project") == "settings-project"
    assert kwargs.get("location") == "settings-location"
