import deepeval.models.llms.local_model as local_mod

from pydantic import SecretStr

from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.llms.local_model import LocalModel
from tests.test_core.stubs import _RecordingClient


def _stub_openai_clients(monkeypatch):
    """Avoid constructing real OpenAI clients in tests."""
    monkeypatch.setattr(local_mod, "OpenAI", _RecordingClient, raising=True)
    monkeypatch.setattr(
        local_mod, "AsyncOpenAI", _RecordingClient, raising=True
    )


##########################
# Test Secret Management #
##########################


def test_local_model_uses_explicit_params_over_settings_and_strips_secret(
    monkeypatch,
):
    """
    Explicit ctor api_key/base_url/model/format must override Settings.*,
    and the client should see a plain string api_key even if Settings
    stores a SecretStr.
    """
    # Seed env so Settings sees baseline values
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "env-secret-key")
    monkeypatch.setenv("LOCAL_MODEL_NAME", "settings-model")
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "http://settings-host:11434/v1")
    monkeypatch.setenv("LOCAL_MODEL_FORMAT", "settings-format")

    # Rebuild Settings from env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity: API key should be a SecretStr on the settings object
    assert isinstance(settings.LOCAL_MODEL_API_KEY, SecretStr)

    # Stub OpenAI clients so we don't make any real calls
    _stub_openai_clients(monkeypatch)

    # Explicit ctor args should override everything from Settings
    model = LocalModel(
        model="ctor-model",
        api_key="ctor-secret-key",
        base_url="http://ctor-host:11434/v1",
        format="ctor-format",
    )

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    kw = client.kwargs

    # Client sees ctor api_key, not Settings.LOCAL_MODEL_API_KEY
    api_key = kw.get("api_key")
    assert isinstance(api_key, str)
    assert api_key == "ctor-secret-key"

    # Base URL should come from ctor as well
    base_url = kw.get("base_url")
    assert base_url is not None
    assert base_url.rstrip("/") == "http://ctor-host:11434/v1"

    # Model attributes reflect ctor overrides
    assert model.model_name == "ctor-model"
    assert model.format == "ctor-format"


def test_local_model_defaults_from_settings(monkeypatch):
    """
    When no ctor args are provided, LocalModel should pull its configuration
    (API key, model name, base_url, format) from Settings, which in turn are
    backed by env vars.
    """
    # Seed env so Settings picks up all Local-related values
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "env-secret-key")
    monkeypatch.setenv("LOCAL_MODEL_NAME", "settings-model")
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "http://settings-host:11434/v1")
    monkeypatch.setenv("LOCAL_MODEL_FORMAT", "settings-format")

    # Rebuild settings from env
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Sanity: API key should be a SecretStr on the settings object
    assert isinstance(settings.LOCAL_MODEL_API_KEY, SecretStr)

    # Stub OpenAI SDK clients so no real network calls happen
    _stub_openai_clients(monkeypatch)

    # No ctor args: everything should come from Settings
    model = LocalModel()

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    kw = client.kwargs

    # Client kwargs pulled from Settings
    assert kw.get("api_key") == "env-secret-key"
    base_url = kw.get("base_url")
    assert base_url is not None
    assert base_url.rstrip("/") == "http://settings-host:11434/v1"

    # Model name and format should also come from Settings
    assert model.model_name == "settings-model"
    assert model.format == "settings-format"


def test_local_model_build_client_unwraps_secret_from_settings(monkeypatch):
    """
    _build_client should unwrap the SecretStr from Settings.LOCAL_MODEL_API_KEY
    (or the stored SecretStr field) into a plain string before passing it to
    the underlying OpenAI client.
    """
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "env-secret-key")
    monkeypatch.setenv("LOCAL_MODEL_NAME", "settings-model")
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "http://settings-host:11434/v1")
    monkeypatch.setenv("LOCAL_MODEL_FORMAT", "settings-format")

    reset_settings(reload_dotenv=False)
    settings = get_settings()
    assert isinstance(settings.LOCAL_MODEL_API_KEY, SecretStr)

    # We don't need to stub OpenAI here because we call _build_client
    # directly with our _RecordingClient stub.
    model = LocalModel()

    # Directly exercise _build_client to verify the kwargs
    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    api_key = kw.get("api_key")
    assert isinstance(api_key, str)
    assert api_key == "env-secret-key"

    base_url = kw.get("base_url")
    assert base_url is not None
    assert base_url.rstrip("/") == "http://settings-host:11434/v1"
