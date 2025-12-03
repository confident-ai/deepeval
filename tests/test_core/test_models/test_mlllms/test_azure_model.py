"""Tests for MultimodalAzureOpenAIMLLMModel settings + secret handling."""

import deepeval.models.mlllms.azure_model as azure_mod

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
        api_key="ctor-secret-key",
        openai_api_version="2099-01-01-preview",
        base_url="https://ctor-endpoint",
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


########################################################
# Legacy keyword backwards compatibility behavior      #
########################################################


def test_multimodal_openai_model_accepts_legacy_azure_endpoint_keyword_and_maps_to_base_url(
    settings,
):
    """
    Using the legacy `model` keyword should still work:
    - It should populate `model_name`
    - It should not be forwarded through `model.kwargs`
    """
    with settings.edit(persist=False):
        settings.AZURE_OPENAI_API_KEY = "test-key"
        settings.OPENAI_API_VERSION = "4.1"

    model = MultimodalAzureOpenAIMLLMModel(azure_endpoint="https://example.com")

    # legacy keyword mapped to canonical parameter
    assert model.base_url == "https://example.com"

    # legacy key should not be forwarded to the client kwargs
    assert "azure_endpoint" not in model.kwargs


def test_multimodal_openai_model_accepts_legacy_api_key_keyword_and_uses_it(
    monkeypatch,
):
    """
    Using the legacy `azure_openai_api_key` keyword should:
    - Populate the canonical `api_key` (via SecretStr)
    - Result in the underlying client receiving the correct `api_key` value
    - Not forward `azure_openai_api_key` in model.kwargs
    """
    # Put AZURE_OPENAI_API_KEY into the process env so Settings sees it
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-secret-key")

    # rebuild the Settings singleton from the current env
    reset_settings(reload_dotenv=False)
    settings = get_settings()
    assert isinstance(settings.AZURE_OPENAI_API_KEY, SecretStr)

    # Stub the Azure SDK clients so we don't make any real calls
    monkeypatch.setattr(
        azure_mod, "AzureOpenAI", _RecordingClient, raising=True
    )
    monkeypatch.setattr(
        azure_mod, "AsyncAzureOpenAI", _RecordingClient, raising=True
    )

    # Construct AzureOpenAIModel with the legacy key name
    model = MultimodalAzureOpenAIMLLMModel(
        model_name="claude-3-7-sonnet-latest",
        azure_openai_api_key="constructor-key",
    )

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    api_key = client.kwargs.get("api_key")

    # The client should see a plain string API key coming from the legacy param
    assert isinstance(api_key, str)
    assert api_key == "constructor-key"

    # And the legacy key should not be present in the model's kwargs
    assert "azure_openai_api_key" not in model.kwargs
