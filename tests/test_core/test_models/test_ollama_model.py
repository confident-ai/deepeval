from unittest.mock import patch

from deepeval.config.settings import get_settings, reset_settings
from deepeval.models.llms.ollama_model import OllamaModel


@patch("deepeval.models.llms.ollama_model.Client")
def test_ollama_model_uses_explicit_model_and_base_url_over_settings(
    mock_client_cls,
):
    """
    Explicit ctor `model` and `base_url` must override Settings-based
    defaults, and the underlying Ollama Client must be constructed with
    the explicit host.
    """
    # Fresh Settings instance
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Seed Settings with default values that *should not* be used
    with settings.edit(persist=False):
        settings.LOCAL_MODEL_NAME = "settings-model"
        settings.LOCAL_MODEL_BASE_URL = "http://settings-host:11434"

    # Instantiate with explicit overrides
    model = OllamaModel(
        model="ctor-model",
        base_url="http://ctor-host:11434",
    )

    # DeepEvalBaseLLM.__init__ calls load_model(), which should call Client(...)
    mock_client_cls.assert_called_once()
    _, kwargs = mock_client_cls.call_args

    # Client must see the ctor host, and model_name must be the ctor model
    assert kwargs.get("host") == "http://ctor-host:11434"
    assert model.model_name == "ctor-model"


@patch("deepeval.models.llms.ollama_model.Client")
def test_ollama_model_defaults_model_and_base_url_from_settings(
    mock_client_cls,
):
    """
    When no ctor `model` or `base_url` is provided, OllamaModel should
    resolve both values from the Pydantic Settings object
    (LOCAL_MODEL_NAME, LOCAL_MODEL_BASE_URL), and construct the Client
    with that host.
    """
    # Fresh Settings instance
    reset_settings(reload_dotenv=False)
    settings = get_settings()

    # Seed Settings with the values that should be used by default
    with settings.edit(persist=False):
        settings.LOCAL_MODEL_NAME = "settings-model"
        settings.LOCAL_MODEL_BASE_URL = "http://settings-host:11434"

    # No ctor overrides: everything should come from Settings
    model = OllamaModel()

    # DeepEvalBaseLLM.__init__ calls load_model(), which should call Client(...)
    mock_client_cls.assert_called_once()
    _, kwargs = mock_client_cls.call_args

    # Model name and host must match the Settings values (ignoring trailing slash normalization)
    assert model.model_name == "settings-model"
    host = kwargs.get("host")
    assert host is not None
    assert host.rstrip("/") == "http://settings-host:11434"
