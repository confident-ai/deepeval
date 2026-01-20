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


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_vertexai_allows_adc_when_no_service_account_key(
    mock_require_dep,
    settings,
):
    """
    Vertex AI mode should allow Application Default Credentials (ADC)

    With GOOGLE_GENAI_USE_VERTEXAI enabled and project/location set,
    GeminiModel should create a Vertex client even when no service account
    key is provided. In that case, credentials should be None and resolved via ADC.
    """
    fake_genai = _make_fake_genai_module()

    def _fake_require_dependency(name, *args, **kwargs):
        # ADC path should only need the genai module and not require oauth2
        # just to allow default creds.
        if name == "google.genai":
            return fake_genai
        raise AssertionError(f"Unexpected dependency requested: {name}")

    mock_require_dep.side_effect = _fake_require_dependency

    with settings.edit(persist=False):
        settings.GOOGLE_GENAI_USE_VERTEXAI = True
        settings.GOOGLE_CLOUD_PROJECT = "test-project"
        settings.GOOGLE_CLOUD_LOCATION = "us-central1"
        settings.GOOGLE_SERVICE_ACCOUNT_KEY = None

    model = GeminiModel(
        model="gemini-1.5-pro",
        project="test-project",
        location="us-central1",
        service_account_key=None,
    )

    client = model.model

    # assert that we are building a Vertex client rather than API-key mode
    assert client.kwargs.get("vertexai") is True
    assert client.kwargs.get("project") == "test-project"
    assert client.kwargs.get("location") == "us-central1"

    # credentials should be absent/None so the SDK resolves via ADC.
    assert client.kwargs.get("credentials") is None


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_model_use_vertexai_param_overrides_settings(
    mock_require_dep,
    settings,
):
    """
    Explicit ctor `use_vertexai` must override Settings.GOOGLE_GENAI_USE_VERTEXAI,
    including when explicitly set to False.
    """
    fake_genai = _make_fake_genai_module()

    def _fake_require_dependency(name, *args, **kwargs):
        if name == "google.genai":
            return fake_genai
        raise AssertionError(f"Unexpected dependency requested: {name}")

    mock_require_dep.side_effect = _fake_require_dependency

    # Case 1: settings says True, ctor forces False -> API-key client
    with settings.edit(persist=False):
        settings.GOOGLE_GENAI_USE_VERTEXAI = True
        settings.GOOGLE_API_KEY = "env-secret-key"
        # even if these are set, we should NOT use Vertex due to ctor override
        settings.GOOGLE_CLOUD_PROJECT = "test-project"
        settings.GOOGLE_CLOUD_LOCATION = "us-central1"

    model = GeminiModel(
        model="gemini-1.5-pro",
        use_vertexai=False,
    )
    client = model.model
    assert client.kwargs.get("vertexai") is not True
    assert client.kwargs.get("api_key") == "env-secret-key"

    # Case 2: settings says False, ctor forces True -> Vertex client
    with settings.edit(persist=False):
        settings.GOOGLE_GENAI_USE_VERTEXAI = False
        settings.GOOGLE_CLOUD_PROJECT = "test-project"
        settings.GOOGLE_CLOUD_LOCATION = "us-central1"
        settings.GOOGLE_SERVICE_ACCOUNT_KEY = None

    model = GeminiModel(
        model="gemini-1.5-pro",
        use_vertexai=True,
        project="test-project",
        location="us-central1",
        service_account_key=None,
    )
    client = model.model
    assert client.kwargs.get("vertexai") is True
    assert client.kwargs.get("project") == "test-project"
    assert client.kwargs.get("location") == "us-central1"
    assert client.kwargs.get("credentials") is None
