import pytest

import deepeval.models.llms.anthropic_model as anthropic_mod
from deepeval.errors import DeepEvalError
from deepeval.models.llms.anthropic_model import AnthropicModel
from deepeval.config.settings import get_settings, reset_settings
from pydantic import SecretStr

from tests.test_core.stubs import _RecordingClient


def test_anthropic_model_uses_explicit_key_over_settings_and_strips_secret(
    monkeypatch,
):
    """
    Added with fix for Issue: #2326
    """
    # Put ANTHROPIC_API_KEY into the process env so Settings sees it
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-secret-key")

    # rebuild the Settings singleton from the current env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity check: Settings should expose this as a SecretStr
    assert isinstance(settings.ANTHROPIC_API_KEY, SecretStr)

    # Stub the Anthropic SDK clients so we don't make any real calls
    monkeypatch.setattr(
        anthropic_mod, "Anthropic", _RecordingClient, raising=True
    )
    monkeypatch.setattr(
        anthropic_mod, "AsyncAnthropic", _RecordingClient, raising=True
    )

    # Construct AnthropicModel with an explicit key
    model = AnthropicModel(
        model="claude-3-7-sonnet-latest",
        _anthropic_api_key="constructor-key",
    )

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    api_key = client.kwargs.get("api_key")

    # Before the fix for Issue #2326:
    #   api_key is the SecretStr from settings.ANTHROPIC_API_KEY, and this assertion FAILS.
    # After the fix:
    #   api_key is a plain str, equal to the explicit constructor key.
    assert isinstance(api_key, str)
    assert api_key == "constructor-key"


def test_anthropic_model_uses_settings_key_when_no_explicit_key(monkeypatch):
    # Ensure env has a key
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-only-key")
    reset_settings(reload_dotenv=False)
    settings = get_settings()
    assert isinstance(settings.ANTHROPIC_API_KEY, SecretStr)

    # Stub Anthropic client to avoid real network and inspect kwargs
    monkeypatch.setattr(
        anthropic_mod, "Anthropic", _RecordingClient, raising=True
    )

    model = AnthropicModel(model="claude-3-7-sonnet-latest")
    client = model.model
    assert client.kwargs["api_key"] == "env-only-key"


def test_anthropic_model_uses_explicit_key_when_settings_missing(monkeypatch):
    # Make sure ANTHROPIC_API_KEY is not present
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings(reload_dotenv=False)
    settings = get_settings()
    assert settings.ANTHROPIC_API_KEY is None

    monkeypatch.setattr(
        anthropic_mod, "Anthropic", _RecordingClient, raising=True
    )

    model = AnthropicModel(
        model="claude-3-7-sonnet-latest",
        _anthropic_api_key="explicit-key",
    )
    client = model.model
    assert client.kwargs["api_key"] == "explicit-key"


def test_anthropic_model_raises_when_no_key_configured(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings(reload_dotenv=False)
    settings = get_settings()
    assert settings.ANTHROPIC_API_KEY is None

    monkeypatch.setattr(
        anthropic_mod, "Anthropic", _RecordingClient, raising=True
    )

    with pytest.raises(DeepEvalError, match="not configured"):
        AnthropicModel(model="claude-3-7-sonnet-latest")


def test_anthropic_model_raises_when_explicit_key_empty(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings(reload_dotenv=False)

    monkeypatch.setattr(
        anthropic_mod, "Anthropic", _RecordingClient, raising=True
    )

    with pytest.raises(DeepEvalError, match="empty"):
        AnthropicModel(
            model="claude-3-7-sonnet-latest",
            _anthropic_api_key="",
        )


def test_anthropic_model_raises_when_settings_key_empty(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    reset_settings(reload_dotenv=False)
    settings = get_settings()
    # pydantic will treat this as SecretStr(""), which is what we want to test
    assert isinstance(settings.ANTHROPIC_API_KEY, SecretStr)
    assert settings.ANTHROPIC_API_KEY.get_secret_value() == ""

    monkeypatch.setattr(
        anthropic_mod, "Anthropic", _RecordingClient, raising=True
    )

    with pytest.raises(DeepEvalError, match="empty"):
        AnthropicModel(model="claude-3-7-sonnet-latest")
