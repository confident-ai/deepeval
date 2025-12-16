from pydantic import SecretStr

import deepeval.models.embedding_models.azure_embedding_model as azure_mod

from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.embedding_models.azure_embedding_model import (
    AzureOpenAIEmbeddingModel,
)
from tests.test_core.stubs import _RecordingClient


##########################
# Test Secret Management #
##########################


def test_azure_embedding_model_uses_explicit_params_over_settings_and_strips_secret(
    monkeypatch,
):
    """
    Explicit ctor args (openai_api_key / version / endpoint / deployment / model)
    must override Settings.*, and _build_client should see a plain string API key
    even though Settings stores a SecretStr.
    """
    # Seed env so Settings sees baseline Azure values
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-secret-key")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-02-15-preview")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://settings-endpoint")
    monkeypatch.setenv(
        "AZURE_EMBEDDING_DEPLOYMENT_NAME", "settings-embed-deployment"
    )

    # Rebuild the Settings singleton from the current env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Settings should expose the API key as a SecretStr
    assert isinstance(settings.AZURE_OPENAI_API_KEY, SecretStr)

    # Explicit ctor args should override everything from Settings
    model = AzureOpenAIEmbeddingModel(
        api_key="ctor-secret-key",
        api_version="2099-01-01-preview",
        base_url="https://ctor-endpoint",
        deployment_name="ctor-deployment",
        model="ctor-model",
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
    assert kw.get("azure_endpoint") == "https://ctor-endpoint"
    assert kw.get("azure_deployment") == "ctor-deployment"

    # Model name should match the ctor-provided model
    assert model.name == "ctor-model"


def test_azure_embedding_model_defaults_from_settings(monkeypatch):
    """
    When no ctor args are provided, AzureOpenAIEmbeddingModel should pull its
    configuration (API key, version, endpoint, deployment) from Settings,
    which in turn is backed by env vars.
    """
    # Seed env so Settings picks up all Azure-related values
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-secret-key")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-02-15-preview")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://settings-endpoint")
    monkeypatch.setenv(
        "AZURE_EMBEDDING_DEPLOYMENT_NAME", "settings-embed-deployment"
    )

    # Rebuild settings from env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # API key should be a SecretStr on the settings object
    assert isinstance(settings.AZURE_OPENAI_API_KEY, SecretStr)

    # No ctor args: everything should come from Settings
    model = AzureOpenAIEmbeddingModel()

    # Directly exercise _build_client to verify the resolved kwargs
    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    # Client kwargs pulled from Settings
    api_key = kw.get("api_key")
    assert isinstance(api_key, str)
    assert api_key == "env-secret-key"

    assert kw.get("api_version") == "2024-02-15-preview"

    endpoint = kw.get("azure_endpoint")
    assert endpoint is not None
    # Allow trailing slash differences
    assert endpoint.rstrip("/") == "https://settings-endpoint"

    assert kw.get("azure_deployment") == "settings-embed-deployment"

    # Model name should default to the Azure embedding deployment
    assert model.name == "settings-embed-deployment"


########################################################
# Legacy keyword backwards compatibility behavior      #
########################################################


def test_azure_embedding_model_accepts_legacy_azure_endpoint_keyword_and_maps_to_base_url(
    settings,
):
    """
    Using the legacy `model` keyword should still work:
    - It should populate `model`
    - It should not be forwarded through `model.kwargs`
    """
    with settings.edit(persist=False):
        settings.AZURE_OPENAI_API_KEY = "test-key"
        settings.OPENAI_API_VERSION = "4.1"

    model = AzureOpenAIEmbeddingModel(base_url="https://example.com")

    # legacy keyword mapped to canonical parameter
    assert model.base_url == "https://example.com"

    # legacy key should not be forwarded to the client kwargs
    assert "azure_endpoint" not in model.kwargs


def test_azure_embedding_model_accepts_legacy_api_key_keyword_and_uses_it(
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
    model = AzureOpenAIEmbeddingModel(
        model="claude-3-7-sonnet-latest",
        api_key="constructor-key",
    )

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    api_key = client.kwargs.get("api_key")

    # The client should see a plain string API key coming from the legacy param
    assert isinstance(api_key, str)
    assert api_key == "constructor-key"

    # And the legacy key should not be present in the model's kwargs
    assert "openai_api_key" not in model.kwargs
