from unittest.mock import patch, MagicMock

from deepeval.models.llms.groq_model import GroqModel

##########################
# Test Secret Management #
##########################


def _make_fake_groq_module():
    """
    Creates a mock module to simulate the 'groq' library.
    This prevents the test from failing if the user hasn't pip installed groq.
    """
    mock_groq_module = MagicMock()
    # We mock the Groq class so we can intercept its initialization
    mock_groq_module.Groq = MagicMock()
    return mock_groq_module


@patch("deepeval.models.llms.groq_model.require_dependency")
def test_groq_model_uses_explicit_key_over_settings_and_passes_plain_str(
    mock_require_dep,
    settings,
):
    """
    Explicit ctor `api_key` must override Settings.GROQ_API_KEY, and the
    underlying Client must see a plain string (not SecretStr).
    """
    fake_groq = _make_fake_groq_module()
    mock_require_dep.return_value = fake_groq

    # Seed env so Settings sees GROQ_API_KEY
    with settings.edit(persist=False):
        settings.GROQ_API_KEY = "env-secret-key"

    # Construct with an explicit api_key – this must win over Settings
    model = GroqModel(
        model="llama3-8b-8192",
        api_key="ctor-secret-key",
    )

    # Trigger the lazy-loading of the client
    model.load_model()

    # Verify the underlying Groq client was called
    fake_groq.Groq.assert_called_once()

    # Extract the arguments passed to the Groq() constructor
    _, kwargs = fake_groq.Groq.call_args
    api_key = kwargs.get("api_key")

    # Client must see the ctor key, as a plain string
    assert isinstance(api_key, str)
    assert api_key == "ctor-secret-key"


@patch("deepeval.models.llms.groq_model.require_dependency")
def test_groq_model_defaults_key_from_settings_and_unwraps_secret(
    mock_require_dep,
    settings,
):
    """
    When no ctor `api_key` is provided, GroqModel should pull the key
    from Settings.GROQ_API_KEY and unwrap it to a plain string for the
    underlying Client.
    """
    fake_groq = _make_fake_groq_module()
    mock_require_dep.return_value = fake_groq

    # Seed env so Settings picks up GROQ_API_KEY
    with settings.edit(persist=False):
        settings.GROQ_API_KEY = "env-secret-key"

    # No ctor api_key, it must come from Settings.GROQ_API_KEY
    model = GroqModel(
        model="llama3-8b-8192",
    )

    # Trigger the lazy-loading of the client
    model.load_model()

    # Verify the underlying Groq client was called
    fake_groq.Groq.assert_called_once()

    # Extract the arguments passed to the Groq() constructor
    _, kwargs = fake_groq.Groq.call_args
    api_key = kwargs.get("api_key")

    # Client must see the Settings key, as a plain string
    assert isinstance(api_key, str)
    assert api_key == "env-secret-key"
