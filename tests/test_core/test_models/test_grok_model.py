"""Tests for GrokModel settings + secret handling (GROK_* + TEMPERATURE)."""

import deepeval.models.llms.grok_model as grok_mod

from pydantic import SecretStr

from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.llms.grok_model import GrokModel
from tests.test_core.stubs import _RecordingClient


def _stub_load_model(monkeypatch):
    """Avoid importing xai_sdk in tests by stubbing load_model."""

    def fake_load_model(self, async_mode: bool = False):
        return _RecordingClient()

    monkeypatch.setattr(
        grok_mod.GrokModel,
        "load_model",
        fake_load_model,
        raising=True,
    )


#####################################
# API key / model name / temperature
#####################################


def test_grok_model_uses_explicit_key_over_settings_and_strips_secret(
    monkeypatch,
):
    """
    Explicit ctor api_key must override Settings.GROK_API_KEY, and the client
    should see a plain string, even if Settings stores a SecretStr.
    """
    # Seed env so Settings sees a GROK_API_KEY
    monkeypatch.setenv("GROK_API_KEY", "env-secret-key")
    monkeypatch.setenv("GROK_MODEL_NAME", "grok-3")

    # Rebuild Settings from env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity: Settings should expose this as SecretStr
    assert isinstance(settings.GROK_API_KEY, SecretStr)

    # Prevent __init__ from importing xai_sdk
    _stub_load_model(monkeypatch)

    # ctor api_key should win over Settings.GROK_API_KEY
    model = GrokModel(
        model="grok-3",
        api_key="ctor-secret-key",
    )

    # _build_client should unwrap the SecretStr to a plain string
    client = model._build_client(_RecordingClient)
    api_key = client.kwargs.get("api_key")

    assert isinstance(api_key, str)
    assert api_key == "ctor-secret-key"


def test_grok_model_defaults_from_settings(monkeypatch):
    """
    When no ctor args are provided, GrokModel should pull model/api_key from
    Settings, which are backed by env vars.
    """
    monkeypatch.setenv("GROK_API_KEY", "env-secret-key")
    monkeypatch.setenv("GROK_MODEL_NAME", "grok-3")

    reset_settings(reload_dotenv=False)
    settings = get_settings()

    assert isinstance(settings.GROK_API_KEY, SecretStr)

    _stub_load_model(monkeypatch)

    # No ctor args: everything should come from Settings
    model = GrokModel()

    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    # Client sees the env/Settings value
    assert kw.get("api_key") == "env-secret-key"
    # Model name from Settings
