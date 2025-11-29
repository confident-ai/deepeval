"""Tests for AzureOpenAIModel generation_kwargs parameter"""

import deepeval.models.llms.azure_model as azure_mod

from unittest.mock import Mock, patch
from pydantic import BaseModel, SecretStr
import pytest
from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.llms.azure_model import AzureOpenAIModel
from tests.test_core.stubs import _RecordingClient


class SampleSchema(BaseModel):
    """Sample schema for structured output testing"""

    field1: str
    field2: int


class TestAzureOpenAIModelGenerationKwargs:
    """Test suite for AzureOpenAIModel generation_kwargs functionality"""

    def test_init_without_generation_kwargs(self, settings):
        """Test that AzureOpenAIModel initializes correctly without generation_kwargs"""
        with settings.edit(persist=False):
            settings.AZURE_OPENAI_API_KEY = "test-key"
            settings.AZURE_OPENAI_ENDPOINT = "http://test-endpoint"
            settings.AZURE_DEPLOYMENT_NAME = "test-deployment"
            settings.AZURE_MODEL_NAME = "gpt-4"
            settings.OPENAI_API_VERSION = "2024-02-15-preview"

        model = AzureOpenAIModel()
        assert model.generation_kwargs == {}
        assert model.kwargs == {}

    def test_init_with_generation_kwargs(self, settings):
        """Test that AzureOpenAIModel initializes correctly with generation_kwargs"""
        with settings.edit(persist=False):
            settings.AZURE_OPENAI_API_KEY = "test-key"
            settings.AZURE_OPENAI_ENDPOINT = "http://test-endpoint"
            settings.AZURE_DEPLOYMENT_NAME = "test-deployment"
            settings.AZURE_MODEL_NAME = "gpt-4"
            settings.OPENAI_API_VERSION = "2024-02-15-preview"

        generation_kwargs = {
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
        }
        model = AzureOpenAIModel(generation_kwargs=generation_kwargs)
        assert model.generation_kwargs == generation_kwargs
        assert model.kwargs == {}

    def test_init_with_both_client_and_generation_kwargs(self, settings):
        """Test that client kwargs and generation_kwargs are kept separate"""

        with settings.edit(persist=False):
            settings.AZURE_OPENAI_API_KEY = "test-key"
            settings.AZURE_OPENAI_ENDPOINT = "http://test-endpoint"
            settings.AZURE_DEPLOYMENT_NAME = "test-deployment"
            settings.AZURE_MODEL_NAME = "gpt-4"
            settings.OPENAI_API_VERSION = "2024-02-15-preview"

        generation_kwargs = {"max_tokens": 500}
        model = AzureOpenAIModel(
            generation_kwargs=generation_kwargs,
            timeout=30,  # client kwarg
            max_retries=3,  # client kwarg
        )
        assert model.generation_kwargs == generation_kwargs
        assert model.kwargs == {"timeout": 30, "max_retries": 3}

    @patch("deepeval.models.llms.azure_model.AzureOpenAIModel.load_model")
    def test_generate_with_generation_kwargs(self, mock_load_model):
        """Test that generation_kwargs are passed to generate method"""
        # Setup mock
        mock_client = Mock()
        mock_load_model.return_value = mock_client
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="test response"))]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_completion

        # Create model with explicit deployment_name
        model = AzureOpenAIModel(
            deployment_name="test-deployment",
            model_name="gpt-4",
            azure_openai_api_key="test-key",
            azure_endpoint="test-endpoint",
            openai_api_version="2024-02-15-preview",
            generation_kwargs={"max_tokens": 1000, "top_p": 0.9},
        )

        # Call generate
        output, cost = model.generate("test prompt")

        # Verify the completion was called with generation_kwargs
        mock_client.chat.completions.create.assert_called_once_with(
            model="test-deployment",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0,
            max_tokens=1000,
            top_p=0.9,
        )
        assert output == "test response"

    @patch("deepeval.models.llms.azure_model.AzureOpenAIModel.load_model")
    def test_generate_without_generation_kwargs(self, mock_load_model):
        """Test that generate works without generation_kwargs"""
        # Setup mock
        mock_client = Mock()
        mock_load_model.return_value = mock_client
        mock_completion = Mock()
        mock_completion.choices = [Mock(message=Mock(content="test response"))]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_completion

        # Create model with explicit deployment_name
        model = AzureOpenAIModel(
            deployment_name="test-deployment",
            model_name="gpt-4",
            azure_openai_api_key="test-key",
            azure_endpoint="test-endpoint",
            openai_api_version="2024-02-15-preview",
        )

        # Call generate without generation_kwargs
        output, cost = model.generate("test prompt")

        # Verify the completion was called without extra kwargs
        mock_client.chat.completions.create.assert_called_once_with(
            model="test-deployment",
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0,
        )
        assert output == "test response"

    @patch("deepeval.models.llms.azure_model.AzureOpenAI")
    def test_load_model_passes_kwargs_to_client(self, mock_azure_openai):
        """Test that client kwargs are passed, and SDK retries are disabled"""
        mock_client = Mock()
        mock_azure_openai.return_value = mock_client

        model = AzureOpenAIModel(
            deployment_name="test-deployment",
            model_name="gpt-4",
            azure_openai_api_key="test-key",
            azure_endpoint="test-endpoint",
            openai_api_version="2024-02-15-preview",
            timeout=30,
            max_retries=5,  # user-provided, but we should override it to 0
        )

        mock_azure_openai.reset_mock()

        _ = model.load_model(async_mode=False)

        mock_azure_openai.assert_called_once()
        call_kwargs = mock_azure_openai.call_args[1]

        assert call_kwargs["timeout"] == 30
        # deepeval disables SDK retries to avoid double retries (Tenacity handles them)
        assert call_kwargs["max_retries"] == 0

    def test_backwards_compatibility(self, settings):
        """Test that existing code without generation_kwargs still works"""

        with settings.edit(persist=False):
            settings.AZURE_OPENAI_API_KEY = "test-key"
            settings.AZURE_OPENAI_ENDPOINT = "http://test-endpoint"
            settings.AZURE_DEPLOYMENT_NAME = "test-deployment"
            settings.AZURE_MODEL_NAME = "gpt-4"
            settings.OPENAI_API_VERSION = "2024-02-15-preview"

        # This should work exactly as before
        model = AzureOpenAIModel(temperature=0.5, timeout=30)  # client kwarg
        assert model.temperature == 0.5
        assert model.kwargs == {"timeout": 30}
        assert model.generation_kwargs == {}

    def test_empty_generation_kwargs(self, settings):
        """Test that empty generation_kwargs dict works correctly"""
        with settings.edit(persist=False):
            settings.AZURE_OPENAI_API_KEY = "test-key"
            settings.AZURE_OPENAI_ENDPOINT = "http://test-endpoint"
            settings.AZURE_DEPLOYMENT_NAME = "test-deployment"
            settings.AZURE_MODEL_NAME = "gpt-4"
            settings.OPENAI_API_VERSION = "2024-02-15-preview"

        model = AzureOpenAIModel(generation_kwargs={})
        assert model.generation_kwargs == {}

    def test_none_generation_kwargs(self, settings):
        """Test that None generation_kwargs is handled correctly"""
        with settings.edit(persist=False):
            settings.AZURE_OPENAI_API_KEY = "test-key"
            settings.AZURE_OPENAI_ENDPOINT = "http://test-endpoint"
            settings.AZURE_DEPLOYMENT_NAME = "test-deployment"
            settings.AZURE_MODEL_NAME = "gpt-4"
            settings.OPENAI_API_VERSION = "2024-02-15-preview"

        model = AzureOpenAIModel(generation_kwargs=None)
        assert model.generation_kwargs == {}


##########################
# Test Secret Management #
##########################


def test_azure_openai_model_uses_explicit_key_over_settings_and_strips_secret(
    monkeypatch,
):
    # Put AZURE_OPENAI_API_KEY into the process env so Settings sees it
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-secret-key")

    # rebuild the Settings singleton from the current env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity check: Settings should expose this as a SecretStr
    assert isinstance(settings.AZURE_OPENAI_API_KEY, SecretStr)

    # Stub the AzureOpenAi SDK clients so we don't make any real calls
    monkeypatch.setattr(
        azure_mod, "AzureOpenAI", _RecordingClient, raising=True
    )
    monkeypatch.setattr(
        azure_mod, "AsyncAzureOpenAI", _RecordingClient, raising=True
    )

    # Construct the model with an explicit key
    model = AzureOpenAIModel(
        model="gpt-4.1",
        azure_openai_api_key="constructor-key",
    )

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    api_key = client.kwargs.get("api_key")

    assert isinstance(api_key, str)
    assert api_key == "constructor-key"


def test_azure_openai_model_defaults_from_settings(monkeypatch):
    # Seed env so Settings picks up all Azure-related values
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-secret-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example.com")
    monkeypatch.setenv("AZURE_DEPLOYMENT_NAME", "settings-deployment")
    monkeypatch.setenv("AZURE_MODEL_NAME", "settings-model")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-02-15-preview")

    # Rebuild settings from env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity: API key should be a SecretStr on the settings object
    assert isinstance(settings.AZURE_OPENAI_API_KEY, SecretStr)

    # Stub Azure SDK clients so no real network calls happen
    monkeypatch.setattr(
        azure_mod, "AzureOpenAI", _RecordingClient, raising=True
    )
    monkeypatch.setattr(
        azure_mod, "AsyncAzureOpenAI", _RecordingClient, raising=True
    )

    # No ctor args: everything should come from Settings
    model = AzureOpenAIModel()

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    kw = client.kwargs

    # Client kwargs pulled from Settings
    assert kw.get("api_key") == "env-secret-key"
    endpoint = kw.get("azure_endpoint")
    assert endpoint is not None
    assert endpoint.rstrip("/") == "https://azure.example.com"
    assert kw.get("azure_deployment") == "settings-deployment"
    assert kw.get("api_version") == "2024-02-15-preview"

    # Model name should also come from Settings
    assert model.model_name == "settings-model"


def test_azure_openai_model_ctor_args_override_settings(monkeypatch):
    # Baseline Settings values
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "settings-secret-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://settings-endpoint")
    monkeypatch.setenv("AZURE_DEPLOYMENT_NAME", "settings-deployment")
    monkeypatch.setenv("AZURE_MODEL_NAME", "settings-model")
    monkeypatch.setenv("OPENAI_API_VERSION", "2024-02-15-preview")

    reset_settings(reload_dotenv=False)

    # Stub SDK clients
    monkeypatch.setattr(
        azure_mod, "AzureOpenAI", _RecordingClient, raising=True
    )
    monkeypatch.setattr(
        azure_mod, "AsyncAzureOpenAI", _RecordingClient, raising=True
    )

    # Explicit ctor args should override everything from Settings
    model = AzureOpenAIModel(
        deployment_name="ctor-deployment",
        model_name="ctor-model",
        azure_openai_api_key="ctor-secret-key",
        openai_api_version="2099-01-01-preview",
        azure_endpoint="https://ctor-endpoint",
    )

    client = model.model
    kw = client.kwargs

    # API key should come from ctor, not Settings
    assert kw.get("api_key") == "ctor-secret-key"
    # Endpoint & deployment from ctor
    assert kw.get("azure_endpoint") == "https://ctor-endpoint"
    assert kw.get("azure_deployment") == "ctor-deployment"
    # API version from ctor
    assert kw.get("api_version") == "2099-01-01-preview"

    # Model name should match ctor value
    assert model.model_name == "ctor-model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
