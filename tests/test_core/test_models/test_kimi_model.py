"""Tests for KimiModel settings + secret handling (MOONSHOT_*)."""

import deepeval.models.llms.kimi_model as kimi_mod

from pydantic import SecretStr

from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.llms.kimi_model import KimiModel
from tests.test_core.stubs import _RecordingClient


def _stub_openai_clients(monkeypatch):
    """Avoid constructing real OpenAI clients in tests."""
    monkeypatch.setattr(kimi_mod, "OpenAI", _RecordingClient, raising=True)
    monkeypatch.setattr(kimi_mod, "AsyncOpenAI", _RecordingClient, raising=True)


##########################
# Test Secret Management #
##########################


def test_kimi_model_uses_explicit_key_over_settings_and_strips_secret(
    monkeypatch,
):
    """
    Explicit ctor api_key must override Settings.MOONSHOT_API_KEY, and the
    client should see a plain string, even if Settings stores a SecretStr.
    """
    # Seed env so Settings sees a MOONSHOT_API_KEY
    monkeypatch.setenv("MOONSHOT_API_KEY", "env-secret-key")
    # Also provide a default model name so __init__ has something valid
    monkeypatch.setenv("MOONSHOT_MODEL_NAME", "moonshot-v1-8k")

    # Rebuild the Settings singleton from the current env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity check: Settings should expose this as a SecretStr
    assert isinstance(settings.MOONSHOT_API_KEY, SecretStr)

    # Stub OpenAI clients so we don't make any real calls
    _stub_openai_clients(monkeypatch)

    # Construct the model with an explicit key
    model = KimiModel(
        model="moonshot-v1-8k",
        api_key="ctor-secret-key",
    )

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    api_key = client.kwargs.get("api_key")

    # Client sees a plain string from the ctor, not the SecretStr
    assert isinstance(api_key, str)
    assert api_key == "ctor-secret-key"


def test_kimi_model_defaults_from_settings(monkeypatch):
    """
    When no ctor args are provided, KimiModel should pull its configuration
    (API key, model name) from Settings, which in turn are backed by env vars.
    """
    # Seed env so Settings picks up all Kimi/Moonshot-related values
    monkeypatch.setenv("MOONSHOT_API_KEY", "env-secret-key")
    monkeypatch.setenv("MOONSHOT_MODEL_NAME", "moonshot-v1-8k")

    # Rebuild settings from env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity: API key should be a SecretStr on the settings object
    assert isinstance(settings.MOONSHOT_API_KEY, SecretStr)

    # Stub OpenAI SDK clients so no real network calls happen
    _stub_openai_clients(monkeypatch)

    # No ctor args: everything should come from Settings
    model = KimiModel()

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    kw = client.kwargs

    # Client kwargs pulled from Settings
    assert kw.get("api_key") == "env-secret-key"
    assert kw.get("base_url") == "https://api.moonshot.cn/v1"

    # Model name should also come from Settings
    assert model.model_name == "moonshot-v1-8k"


def test_kimi_model_ctor_args_override_settings(monkeypatch):
    """
    Explicit ctor args (api_key/model) should override any values coming from
    Settings/environment.
    """
    # Baseline Settings values
    monkeypatch.setenv("MOONSHOT_API_KEY", "settings-secret-key")
    monkeypatch.setenv("MOONSHOT_MODEL_NAME", "moonshot-v1-8k")

    reset_settings(reload_dotenv=False)

    # Stub SDK clients
    _stub_openai_clients(monkeypatch)

    # Explicit ctor args should override everything from Settings
    model = KimiModel(
        api_key="ctor-secret-key",
        model="moonshot-v1-32k",
        temperature=0.5,
    )

    client = model.model
    kw = client.kwargs

    # API key should come from ctor, not Settings
    assert kw.get("api_key") == "ctor-secret-key"
    # Base URL remains the Moonshot endpoint
    assert kw.get("base_url") == "https://api.moonshot.cn/v1"

    # Model name should match ctor value
    assert model.model_name == "moonshot-v1-32k"
    # And the temperature should respect the ctor argument (assuming no
    # TEMPERATURE override from Settings)
    assert model.temperature == 0.5
