from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pydantic import SecretStr

from deepeval.models.llms.constants import GEMINI_MODELS_DATA
from deepeval.models.llms.gemini_model import GeminiModel
from deepeval.models.utils import EvaluationCost
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


#################################
# Cost behavior: token × price  #
#################################


def _build_gemini_model_with_fake_client(
    mock_require_dep, settings, fake_response, model_name="gemini-1.5-pro"
):
    """Wire a GeminiModel whose underlying client returns ``fake_response``."""
    fake_genai = _make_fake_genai_module()
    fake_genai.types.GenerateContentConfig = lambda **kwargs: kwargs

    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    def _fake_require_dependency(name, *args, **kwargs):
        if name == "google.genai":
            return fake_genai
        raise AssertionError(f"Unexpected dependency: {name}")

    mock_require_dep.side_effect = _fake_require_dependency

    with settings.edit(persist=False):
        settings.GOOGLE_API_KEY = "test-key"

    model = GeminiModel(model=model_name)
    model.load_model = lambda *a, **kw: fake_client
    return model


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_generate_computes_cost_from_tokens_and_registry_prices(
    mock_require_dep, settings
):
    """
    With populated ``usage_metadata`` and a model present in the registry,
    ``generate`` must return an ``EvaluationCost`` whose float value equals
    ``input_tokens × input_price + output_tokens × output_price`` and whose
    ``input_tokens`` / ``output_tokens`` attrs reflect what the SDK reported.
    """
    fake_response = SimpleNamespace(
        text="Hello world",
        usage_metadata=SimpleNamespace(
            prompt_token_count=1000,
            candidates_token_count=500,
        ),
    )

    model = _build_gemini_model_with_fake_client(
        mock_require_dep, settings, fake_response, model_name="gemini-1.5-pro"
    )

    output, cost = model.generate("test prompt")

    registry = GEMINI_MODELS_DATA.get("gemini-1.5-pro")
    expected = 1000 * registry.input_price + 500 * registry.output_price

    assert output == "Hello world"
    assert isinstance(cost, EvaluationCost)
    assert cost == expected
    assert cost > 0  # guard against regressing back to the literal-zero bug
    assert cost.input_tokens == 1000
    assert cost.output_tokens == 500


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_generate_returns_zero_cost_when_usage_metadata_missing(
    mock_require_dep, settings
):
    """
    When the SDK response omits ``usage_metadata``, cost falls back to 0 and
    the token attributes are ``None`` (we cannot invent a price-bearing count).
    """
    fake_response = MagicMock(spec=["text"])
    fake_response.text = "Hello world"

    model = _build_gemini_model_with_fake_client(
        mock_require_dep, settings, fake_response
    )

    output, cost = model.generate("test prompt")

    assert output == "Hello world"
    assert isinstance(cost, EvaluationCost)
    assert cost == 0
    assert cost.input_tokens is None
    assert cost.output_tokens is None


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_generate_returns_zero_cost_for_unregistered_model(
    mock_require_dep, settings
):
    """
    Unknown/custom model names resolve to a default ``DeepEvalModelData`` with
    ``input_price=None`` / ``output_price=None``. Cost must be 0 and must not
    raise, but token counts should still ride along on the EvaluationCost.
    """
    fake_response = SimpleNamespace(
        text="Hello world",
        usage_metadata=SimpleNamespace(
            prompt_token_count=42,
            candidates_token_count=7,
        ),
    )

    model = _build_gemini_model_with_fake_client(
        mock_require_dep,
        settings,
        fake_response,
        model_name="gemini-unregistered-experimental",
    )
    # Sanity: the registry really did fall back to a no-price default.
    assert model.model_data.input_price is None
    assert model.model_data.output_price is None

    output, cost = model.generate("test prompt")

    assert output == "Hello world"
    assert isinstance(cost, EvaluationCost)
    assert cost == 0
    assert cost.input_tokens == 42
    assert cost.output_tokens == 7


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_calculate_cost_unit(mock_require_dep, settings):
    """Direct unit test for ``calculate_cost`` — registry hit and miss."""
    mock_require_dep.return_value = _make_fake_genai_module()

    with settings.edit(persist=False):
        settings.GOOGLE_API_KEY = "test-key"

    model = GeminiModel(model="gemini-2.5-flash")
    registry = GEMINI_MODELS_DATA.get("gemini-2.5-flash")
    expected = 10_000 * registry.input_price + 2_500 * registry.output_price

    result = model.calculate_cost(10_000, 2_500)
    assert isinstance(result, EvaluationCost)
    assert result == expected
    assert result.input_tokens == 10_000
    assert result.output_tokens == 2_500

    # Pricing missing -> contract is to return None (matches OpenAI/Anthropic).
    model.model_data.input_price = None
    assert model.calculate_cost(10_000, 2_500) is None
