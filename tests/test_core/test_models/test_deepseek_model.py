"""Tests for DeepSeekModel generation_kwargs and settings/secret handling."""

import deepeval.models.llms.deepseek_model as deepseek_mod

from unittest.mock import Mock, patch
import pytest
from pydantic import BaseModel, SecretStr

from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.llms.deepseek_model import DeepSeekModel
from tests.test_core.stubs import _RecordingClient


class SampleSchema(BaseModel):
    name: str
    value: int


class TestDeepSeekModelGenerationKwargs:
    def test_init_with_generation_kwargs(self):
        """DeepSeekModel should store generation_kwargs when provided."""
        model = DeepSeekModel(
            api_key="test-key",
            model="deepseek-chat",
            generation_kwargs={"top_p": 0.9, "max_tokens": 123},
        )
        assert model.generation_kwargs == {"top_p": 0.9, "max_tokens": 123}

    def test_init_without_generation_kwargs(self):
        """DeepSeekModel should default generation_kwargs to an empty dict."""
        model = DeepSeekModel(
            api_key="test-key",
            model="deepseek-chat",
            generation_kwargs=None,
        )
        assert model.generation_kwargs == {}

    @patch("deepeval.models.llms.deepseek_model.OpenAI")
    def test_generate_uses_generation_kwargs(self, mock_openai):
        """generation_kwargs should be forwarded into chat.completions.create()."""
        mock_client = Mock()
        mock_completion = Mock()
        # Shape the completion object the way DeepSeekModel expects
        mock_choice = Mock()
        mock_choice.message.content = "hello from deepseek"
        mock_completion.choices = [mock_choice]
        mock_completion.usage.prompt_tokens = 10
        mock_completion.usage.completion_tokens = 5

        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        model = DeepSeekModel(
            api_key="test-key",
            model="deepseek-chat",
            generation_kwargs={"top_p": 0.9},
        )

        output, cost = model.generate("hi there")

        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args

        assert kwargs["model"] == "deepseek-chat"
        assert kwargs["messages"] == [
            {"role": "user", "content": "hi there"},
        ]
        # Our extra kwargs should be preserved
        assert kwargs["top_p"] == 0.9
        # Sanity check on return path
        assert output == "hello from deepseek"
        assert isinstance(cost, (int, float))


##########################
# Test Secret Management #
##########################


def test_deepseek_model_uses_explicit_key_over_settings_and_strips_secret(
    monkeypatch,
):
    """
    Explicit ctor api_key must override Settings.DEEPSEEK_API_KEY, and the
    client should see a plain string, even if Settings stores a SecretStr.
    """
    # Put DEEPSEEK_API_KEY into the process env so Settings sees it
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-secret-key")

    # Rebuild the Settings singleton from the current env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity check: Settings should expose this as a SecretStr
    assert isinstance(settings.DEEPSEEK_API_KEY, SecretStr)

    # Stub the DeepSeek/OpenAI SDK clients so we don't make any real calls
    monkeypatch.setattr(deepseek_mod, "OpenAI", _RecordingClient, raising=True)
    monkeypatch.setattr(
        deepseek_mod, "AsyncOpenAI", _RecordingClient, raising=True
    )

    # Construct the model with an explicit key
    model = DeepSeekModel(
        model="deepseek-chat",
        api_key="ctor-secret-key",
    )

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    api_key = client.kwargs.get("api_key")

    # Client sees a plain string from the ctor, not the SecretStr
    assert isinstance(api_key, str)
    assert api_key == "ctor-secret-key"


def test_deepseek_model_defaults_from_settings(monkeypatch):
    """
    When no ctor args are provided, DeepSeekModel should pull its configuration
    (API key, model name) from Settings, which in turn are backed by env vars.
    """
    # Seed env so Settings picks up all DeepSeek-related values
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-secret-key")
    monkeypatch.setenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")

    # Rebuild settings from env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity: API key should be a SecretStr on the settings object
    assert isinstance(settings.DEEPSEEK_API_KEY, SecretStr)

    # Stub DeepSeek/OpenAI SDK clients so no real network calls happen
    monkeypatch.setattr(deepseek_mod, "OpenAI", _RecordingClient, raising=True)
    monkeypatch.setattr(
        deepseek_mod, "AsyncOpenAI", _RecordingClient, raising=True
    )

    # No ctor args: everything should come from Settings
    model = DeepSeekModel()

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    kw = client.kwargs

    # Client kwargs pulled from Settings
    assert kw.get("api_key") == "env-secret-key"
    assert kw.get("base_url") == "https://api.deepseek.com"

    # Model name should also come from Settings
    assert model.model_name == "deepseek-chat"


def test_deepseek_model_ctor_args_override_settings(monkeypatch):
    """
    Explicit ctor args (api_key/model) should override any values coming from
    Settings/environment.
    """
    # Baseline Settings values
    monkeypatch.setenv("DEEPSEEK_API_KEY", "settings-secret-key")
    monkeypatch.setenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")

    reset_settings(reload_dotenv=False)

    # Stub SDK clients
    monkeypatch.setattr(deepseek_mod, "OpenAI", _RecordingClient, raising=True)
    monkeypatch.setattr(
        deepseek_mod, "AsyncOpenAI", _RecordingClient, raising=True
    )

    # Explicit ctor args should override everything from Settings
    model = DeepSeekModel(
        api_key="ctor-secret-key",
        model="deepseek-reasoner",
        temperature=0.5,
    )

    client = model.model
    kw = client.kwargs

    # API key should come from ctor, not Settings
    assert kw.get("api_key") == "ctor-secret-key"
    # Base URL remains the DeepSeek endpoint
    assert kw.get("base_url") == "https://api.deepseek.com"

    # Model name should match ctor value
    assert model.model_name == "deepseek-reasoner"
    # And the temperature should respect the ctor argument
    assert model.temperature == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
