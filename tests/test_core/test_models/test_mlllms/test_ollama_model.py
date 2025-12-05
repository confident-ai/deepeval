from unittest.mock import patch

from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.mlllms.ollama_model import MultimodalOllamaModel
from tests.test_core.stubs import _RecordingClient, make_fake_ollama_module


@patch("deepeval.models.mlllms.ollama_model.require_dependency")
def test_multimodal_ollama_model_uses_explicit_params_over_settings(
    mock_require_dep,
    settings,
):
    """
    Explicit ctor host/model must override Settings.*, and the underlying
    Ollama client must be constructed with the ctor host even if Settings
    provides defaults.
    """
    # Seed env so Settings sees baseline values
    with settings.edit(persist=False):
        settings.LOCAL_MODEL_BASE_URL = "http://settings-host:11434"
        settings.LOCAL_MODEL_NAME = "settings-llm-model"

    # Rebuild Settings from env
    reset_settings(reload_dotenv=False)
    _ = get_settings()

    # Fake ollama module returned by require_dependency
    fake_ollama = make_fake_ollama_module(_RecordingClient)
    mock_require_dep.return_value = fake_ollama

    # Explicit ctor args should override everything from Settings
    model = MultimodalOllamaModel(
        model="ctor-llm-model",
        host="http://ctor-host:11434",
        timeout=30,  # client kwarg, should pass through to the client
    )

    # exercise load_model() so we go through require_dependency and _build_client
    client = model.load_model()
    kw = client.kwargs

    # Host should come from ctor, not Settings
    host = kw.get("host")
    assert host is not None
    assert host.rstrip("/") == "http://ctor-host:11434"

    # Extra client kwargs should be preserved
    assert kw.get("timeout") == 30

    # Model name should match the ctor-provided model
    assert model.model_name == "ctor-llm-model"

    # Ensure we actually called require_dependency with expected args
    mock_require_dep.assert_any_call(
        "ollama",
        provider_label="MultimodalOllamaModel",
        install_hint="Install it with `pip install ollama`.",
    )


@patch("deepeval.models.mlllms.ollama_model.require_dependency")
def test_multimodal_ollama_model_defaults_from_settings(
    mock_require_dep,
    settings,
):
    """
    When no ctor args are provided, MultimodalOllamaModel should pull host
    and model_name from Settings, which are backed by env vars.
    """
    # Seed env so Settings picks up Ollama related defaults
    with settings.edit(persist=False):
        settings.LOCAL_MODEL_BASE_URL = "http://settings-host:11434"
        settings.LOCAL_MODEL_NAME = "settings-llm-model"

    # Rebuild Settings from env
    reset_settings(reload_dotenv=False)
    _ = get_settings()

    # Fake ollama module returned by require_dependency
    fake_ollama = make_fake_ollama_module(_RecordingClient)
    mock_require_dep.return_value = fake_ollama

    # everything should come from Settings
    model = MultimodalOllamaModel()

    # Exercise load_model() so we go through require_dependency and _build_client
    client = model.load_model()
    kw = client.kwargs

    # Host comes from Settings (allow for trailing slash differences)
    host = kw.get("host")
    assert host is not None
    assert host.rstrip("/") == "http://settings-host:11434"

    # Model name should also come from Settings
    assert model.model_name == "settings-llm-model"

    mock_require_dep.assert_any_call(
        "ollama",
        provider_label="MultimodalOllamaModel",
        install_hint="Install it with `pip install ollama`.",
    )
