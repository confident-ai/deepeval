from deepeval.config.settings import reset_settings
from deepeval.models.mlllms.ollama_model import MultimodalOllamaModel
from tests.test_core.stubs import _RecordingClient


def test_multimodal_ollama_model_uses_explicit_params_over_settings(
    monkeypatch,
):
    """
    Explicit ctor host/model must override Settings.*, and _build_client
    should receive the ctor host even if Settings provides defaults.
    """
    # Seed env so Settings sees baseline values
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "http://settings-host:11434")
    monkeypatch.setenv("LOCAL_MODEL_NAME", "settings-llm-model")

    # Rebuild Settings from env
    reset_settings(reload_dotenv=False)

    # Explicit ctor args should override everything from Settings
    model = MultimodalOllamaModel(
        model_name="ctor-llm-model",
        base_url="http://ctor-host:11434",
        timeout=30,  # client kwarg, should pass through to the client
    )

    # Directly exercise _build_client with our recording stub
    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    # Host should come from ctor, not Settings
    host = kw.get("host")
    assert host is not None
    assert host.rstrip("/") == "http://ctor-host:11434"

    # Extra client kwargs should be preserved
    assert kw.get("timeout") == 30

    # Model name should match the ctor-provided model
    assert model.model_name == "ctor-llm-model"


def test_multimodal_ollama_model_defaults_from_settings(monkeypatch):
    """
    When no ctor args are provided, MultimodalOllamaModel should pull host
    and model_name from Settings, which are backed by env vars.
    """
    # Seed env so Settings picks up Ollama-related defaults
    monkeypatch.setenv("LOCAL_MODEL_BASE_URL", "http://settings-host:11434")
    monkeypatch.setenv("LOCAL_MODEL_NAME", "settings-llm-model")

    # Rebuild Settings from env
    reset_settings(reload_dotenv=False)

    # No ctor args: everything should come from Settings
    model = MultimodalOllamaModel()

    # Directly exercise _build_client with our recording stub
    client = model._build_client(_RecordingClient)
    kw = client.kwargs

    # Host comes from Settings (allow for trailing slash differences)
    host = kw.get("host")
    assert host is not None
    assert host.rstrip("/") == "http://settings-host:11434"

    # Model name should also come from Settings
    assert model.model_name == "settings-llm-model"


########################################################
# Test legacy keyword backwards compatability behavior #
########################################################


def test_multimodal_ollama_model_accepts_legacy_model_keyword_and_maps_to_model_name():
    """
    Using the legacy `model` keyword should still work:
    - It should populate `model_name`
    - It should not be forwarded through `model.kwargs`
    """
    model = MultimodalOllamaModel(model="ctor-model")

    # legacy keyword mapped to canonical parameter
    assert model.model_name == "ctor-model"

    # legacy key should not be forwarded to the client kwargs
    assert "model" not in model.kwargs


def test_multimodal_ollama_model_accepts_legacy_host_keyword_and_maps_to_base_url():
    """
    Using the legacy `model` keyword should still work:
    - It should populate `model_name`
    - It should not be forwarded through `model.kwargs`
    """
    model = MultimodalOllamaModel(host="ctor-host")

    # legacy keyword mapped to canonical parameter
    assert model.base_url == "ctor-host"

    # legacy key should not be forwarded to the client kwargs
    assert "host" not in model.kwargs
