from unittest.mock import patch

from pydantic import SecretStr

from deepeval.models.llms.gemini_model import GeminiModel
from tests.test_core.stubs import _make_fake_genai_module


##########################
# Test Secret Management #
##########################


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_model_uses_explicit_key_over_settings_and_passes_plain_str(
    mock_require_dep,
    settings,
):
    """
    Explicit ctor `api_key` must override Settings.GOOGLE_API_KEY, and the
    underlying Client must see a plain string (not SecretStr).
    """
    # When GeminiModel calls require_dependency(...), return our fake module
    mock_require_dep.return_value = _make_fake_genai_module()

    # Seed env so Settings sees GOOGLE_API_KEY
    with settings.edit(persist=False):
        settings.GOOGLE_API_KEY = "env-secret-key"

    # Settings should expose this as a SecretStr
    assert isinstance(settings.GOOGLE_API_KEY, SecretStr)

    # Construct with an explicit api_key – this must win over Settings
    model = GeminiModel(
        model="gemini-1.5-pro",
        api_key="ctor-secret-key",
    )

    # DeepEvalBaseLLM.__init__ stores the client on `model.model`
    client = model.model
    api_key = client.kwargs.get("api_key")

    # Client must see the ctor key, as a plain string
    assert isinstance(api_key, str)
    assert api_key == "ctor-secret-key"


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_model_defaults_key_from_settings_and_unwraps_secret(
    mock_require_dep,
    settings,
):
    """
    When no ctor `api_key` is provided, GeminiModel should pull the key
    from Settings.GOOGLE_API_KEY and unwrap it to a plain string for the
    underlying Client.
    """
    mock_require_dep.return_value = _make_fake_genai_module()

    # Seed env so Settings picks up GOOGLE_API_KEY
    with settings.edit(persist=False):
        settings.GOOGLE_API_KEY = "env-secret-key"

    # Settings should expose this as a SecretStr
    assert isinstance(settings.GOOGLE_API_KEY, SecretStr)
    assert settings.GOOGLE_API_KEY.get_secret_value() == "env-secret-key"

    # No ctor api_key, it must come from Settings.GOOGLE_API_KEY
    model = GeminiModel(
        model="gemini-1.5-pro",
    )

    client = model.model
    api_key = client.kwargs.get("api_key")

    # Client must see the Settings key, as a plain string
    assert isinstance(api_key, str)
    assert api_key == "env-secret-key"
