import pytest
from types import SimpleNamespace
from unittest.mock import patch

from deepeval.errors import DeepEvalError
from deepeval.models.llms.anthropic_model import AnthropicModel
from deepeval.config.settings import reset_settings, get_settings
from pydantic import SecretStr

from tests.test_core.stubs import _RecordingClient


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_uses_explicit_key_over_settings_and_strips_secret(
    mock_require_dep,
    settings,
):
    """
    Added with fix for Issue: #2326
    """
    # Put ANTHROPIC_API_KEY into the process env so Settings sees it
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = "env-secret-key"

    # rebuild the Settings singleton from the current env
    reset_settings(reload_dotenv=False)

    # Sanity check: Settings should expose this as a SecretStr
    assert isinstance(settings.ANTHROPIC_API_KEY, SecretStr)

    # Fake anthropic module returned by require_dependency
    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

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


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_uses_settings_key_when_no_explicit_key(
    mock_require_dep,
    settings,
):
    # Ensure env has a key
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = "env-only-key"

    reset_settings(reload_dotenv=False)

    assert isinstance(settings.ANTHROPIC_API_KEY, SecretStr)

    # Fake anthropic module returned by require_dependency
    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    # Stub Anthropic client to avoid real network and inspect kwargs
    model = AnthropicModel(model="claude-3-7-sonnet-latest")
    client = model.model
    assert client.kwargs["api_key"] == "env-only-key"


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_uses_explicit_key_when_settings_missing(
    mock_require_dep,
    monkeypatch,
):
    # Make sure ANTHROPIC_API_KEY is not present
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings(reload_dotenv=False)
    settings = get_settings()
    assert settings.ANTHROPIC_API_KEY is None

    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    model = AnthropicModel(
        model="claude-3-7-sonnet-latest",
        _anthropic_api_key="explicit-key",
    )
    client = model.model
    assert client.kwargs["api_key"] == "explicit-key"


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_raises_when_no_key_configured(
    mock_require_dep,
    monkeypatch,
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings(reload_dotenv=False)
    assert get_settings().ANTHROPIC_API_KEY is None

    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    # Error should come from require_secret_api_key / DeepEvalError,
    # not from missing anthropic dependency.
    with pytest.raises(DeepEvalError, match="not configured"):
        AnthropicModel(model="claude-3-7-sonnet-latest")


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_raises_when_explicit_key_empty(
    mock_require_dep,
    monkeypatch,
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reset_settings(reload_dotenv=False)

    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    with pytest.raises(DeepEvalError, match="empty"):
        AnthropicModel(
            model="claude-3-7-sonnet-latest",
            _anthropic_api_key="",
        )


@patch("deepeval.models.llms.anthropic_model.require_dependency")
def test_anthropic_model_raises_when_settings_key_empty(
    mock_require_dep,
    settings,
):
    with settings.edit(persist=False):
        settings.ANTHROPIC_API_KEY = ""
    reset_settings(reload_dotenv=False)
    # pydantic will treat this as SecretStr(""), which is what we want to test
    assert isinstance(settings.ANTHROPIC_API_KEY, SecretStr)
    assert settings.ANTHROPIC_API_KEY.get_secret_value() == ""

    fake_anthropic_module = SimpleNamespace(
        Anthropic=_RecordingClient,
        AsyncAnthropic=_RecordingClient,
    )
    mock_require_dep.return_value = fake_anthropic_module

    with pytest.raises(DeepEvalError, match="empty"):
        AnthropicModel(model="claude-3-7-sonnet-latest")
