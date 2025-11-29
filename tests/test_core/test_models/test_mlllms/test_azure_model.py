"""Tests for MultimodalAzureOpenAIMLLMModel settings + secret handling."""

from pydantic import SecretStr

from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.mlllms.azure_model import MultimodalAzureOpenAIMLLMModel
from tests.test_core.stubs import _RecordingClient


##########################
# Test Settings + Secrets
##########################


def test_multimodal_azure_model_uses_explicit_params_over_settings_and_strips_secret(
    monkeypatch,
):
    """
    Explicit ctor args (deployment/model/api_key/version/endpoint) must override
    Settings.*, and _build_client should see a plain string API key even though
    Settings stores a SecretStr.
    """
    # Seed env so Settings has baseline Azure values
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-secret-key")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-02-15-preview")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://settings-endpoint")
    monkeypatch.setenv("AZURE_MODEL_NAME", "gpt-4o")
    monkeypatch.setenv("AZURE_DEPLOYMENT_NAME", "settings-deployment")

    # Rebuild the Settings singleton from current env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Settings should expose the API key as a SecretStr
    assert isinstance(settings.AZURE_OPENAI_API_KEY, SecretStr)

    # Explicit ctor args should override everything from Settings
    model = MultimodalAzureOpenAIMLLMModel(
        deployment_name="ctor-deployment",
        model_name="gpt-4o-mini",
        azure_openai_api_key="ctor-secret-key",
        openai_api_version="2099-01-01-preview",
        azure_endpoint="https://ctor-endpoint",
    )

    # Directly exercise _build_client with our recording stub
    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    # API key must come from ctor and be a plain string
    api_key = kw.get("api_key")
    assert isinstance(api_key, str)
    assert api_key == "ctor-secret-key"

    # Other ctor params should also be reflected
    assert kw.get("api_version") == "2099-01-01-preview"

    endpoint = kw.get("azure_endpoint")
    assert endpoint is not None
    assert endpoint.rstrip("/") == "https://ctor-endpoint"

    assert kw.get("azure_deployment") == "ctor-deployment"

    # Model name should match the ctor-provided model (after parse_model_name)
    assert model.model_name == "gpt-4o-mini"


def test_multimodal_azure_model_defaults_from_settings(monkeypatch):
    """
    When no ctor args are provided, MultimodalAzureOpenAIMLLMModel should pull
    its configuration (model_name, deployment, API key, version, endpoint)
    from Settings, which in turn is backed by env vars.
    """
    # Seed env so Settings picks up all Azure-related values
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-secret-key")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-02-15-preview")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://settings-endpoint")
    monkeypatch.setenv("AZURE_MODEL_NAME", "gpt-4o")
    monkeypatch.setenv("AZURE_DEPLOYMENT_NAME", "settings-deployment")

    # Rebuild settings from env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # API key should be a SecretStr on the settings object
    assert isinstance(settings.AZURE_OPENAI_API_KEY, SecretStr)

    # No ctor args: everything should come from Settings
    model = MultimodalAzureOpenAIMLLMModel()

    # Directly exercise _build_client to verify resolved kwargs
    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    # Client kwargs pulled from Settings, with API key unwrapped to str
    api_key = kw.get("api_key")
    assert isinstance(api_key, str)
    assert api_key == "env-secret-key"

    assert kw.get("api_version") == "2024-02-15-preview"

    endpoint = kw.get("azure_endpoint")
    assert endpoint is not None
    # Allow for trailing slash differences
    assert endpoint.rstrip("/") == "https://settings-endpoint"

    assert kw.get("azure_deployment") == "settings-deployment"

    # Model name should default from Settings.AZURE_MODEL_NAME (after parse_model_name)
    assert model.model_name == "gpt-4o"
