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
def test_gemini_generate_returns_unknown_cost_when_usage_metadata_missing(
    mock_require_dep, settings
):
    """
    When the SDK response omits ``usage_metadata``, cost is unknown (None) and
    token attributes are also ``None`` (we cannot invent a price-bearing count).
    """
    fake_response = MagicMock(spec=["text"])
    fake_response.text = "Hello world"

    model = _build_gemini_model_with_fake_client(
        mock_require_dep, settings, fake_response
    )

    output, cost = model.generate("test prompt")

    assert output == "Hello world"
    assert isinstance(cost, EvaluationCost)
    assert cost.value is None
    assert cost.input_tokens is None
    assert cost.output_tokens is None


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_generate_returns_unknown_cost_for_unregistered_model(
    mock_require_dep, settings
):
    """
    Unknown/custom model names resolve to a default ``DeepEvalModelData`` with
    ``input_price=None`` / ``output_price=None``. Cost must be None (not 0) and
    must not raise; token counts should still ride along on the EvaluationCost.
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
    assert cost.value is None
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


########################################
# Cost overrides: ctor params + env    #
########################################


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_ctor_cost_params_override_registry(mock_require_dep, settings):
    """
    Explicit ``cost_per_input_token`` / ``cost_per_output_token`` passed to
    the constructor must override whatever the built-in registry says.
    """
    mock_require_dep.return_value = _make_fake_genai_module()

    with settings.edit(persist=False):
        settings.GOOGLE_API_KEY = "test-key"

    custom_input = 0.000001
    custom_output = 0.000002
    model = GeminiModel(
        model="gemini-2.5-flash",
        cost_per_input_token=custom_input,
        cost_per_output_token=custom_output,
    )

    assert model.model_data.input_price == custom_input
    assert model.model_data.output_price == custom_output


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_settings_cost_env_vars_used_as_fallback(
    mock_require_dep, settings
):
    """
    When no ctor cost params are provided, GeminiModel should fall back to
    ``GEMINI_COST_PER_INPUT_TOKEN`` / ``GEMINI_COST_PER_OUTPUT_TOKEN`` from
    settings (i.e. env vars).
    """
    mock_require_dep.return_value = _make_fake_genai_module()

    env_input = 0.0000005
    env_output = 0.0000015
    with settings.edit(persist=False):
        settings.GOOGLE_API_KEY = "test-key"
        settings.GEMINI_COST_PER_INPUT_TOKEN = env_input
        settings.GEMINI_COST_PER_OUTPUT_TOKEN = env_output

    model = GeminiModel(model="gemini-2.5-flash")

    assert model.model_data.input_price == env_input
    assert model.model_data.output_price == env_output


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_ctor_cost_params_take_precedence_over_env(
    mock_require_dep, settings
):
    """
    Explicit ctor cost params must win over ``GEMINI_COST_PER_INPUT_TOKEN``
    / ``GEMINI_COST_PER_OUTPUT_TOKEN`` env vars.
    """
    mock_require_dep.return_value = _make_fake_genai_module()

    with settings.edit(persist=False):
        settings.GOOGLE_API_KEY = "test-key"
        settings.GEMINI_COST_PER_INPUT_TOKEN = 0.0000099
        settings.GEMINI_COST_PER_OUTPUT_TOKEN = 0.0000099

    ctor_input = 0.000001
    ctor_output = 0.000002
    model = GeminiModel(
        model="gemini-2.5-flash",
        cost_per_input_token=ctor_input,
        cost_per_output_token=ctor_output,
    )

    assert model.model_data.input_price == ctor_input
    assert model.model_data.output_price == ctor_output


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_custom_cost_used_in_generate(mock_require_dep, settings):
    """
    End-to-end: a custom cost set via ctor must be reflected in the
    ``EvaluationCost`` returned by ``generate``.
    """
    fake_response = SimpleNamespace(
        text="hi",
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
        ),
    )

    custom_input = 0.000010
    custom_output = 0.000020

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

    model = GeminiModel(
        model="gemini-2.5-flash",
        cost_per_input_token=custom_input,
        cost_per_output_token=custom_output,
    )
    model.load_model = lambda *a, **kw: fake_client

    output, cost = model.generate("test prompt")

    expected = 100 * custom_input + 50 * custom_output
    assert output == "hi"
    assert isinstance(cost, EvaluationCost)
    assert abs(float(cost) - expected) < 1e-12
    assert cost.input_tokens == 100
    assert cost.output_tokens == 50


########################################
# Context-cache discount tests         #
########################################


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_calculate_cost_applies_cache_discount(
    mock_require_dep, settings
):
    """
    When ``cached_tokens > 0`` and the model has ``cache_read_input_price``,
    cached tokens must be charged at the cheaper rate, non-cached at full price.
    """
    mock_require_dep.return_value = _make_fake_genai_module()

    with settings.edit(persist=False):
        settings.GOOGLE_API_KEY = "test-key"

    model = GeminiModel(model="gemini-2.5-flash")
    reg = GEMINI_MODELS_DATA.get("gemini-2.5-flash")
    assert reg.cache_read_input_price is not None, "model must have cache price"

    input_tokens = 1000
    output_tokens = 200
    cached_tokens = 800  # 800 cached, 200 non-cached

    result = model.calculate_cost(input_tokens, output_tokens, cached_tokens)

    expected = (
        (input_tokens - cached_tokens) * reg.input_price
        + cached_tokens * reg.cache_read_input_price
        + output_tokens * reg.output_price
    )
    assert isinstance(result, EvaluationCost)
    assert abs(float(result) - expected) < 1e-15
    assert result.input_tokens == input_tokens
    assert result.output_tokens == output_tokens


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_calculate_cost_zero_cached_tokens_unchanged(
    mock_require_dep, settings
):
    """
    When ``cached_tokens=0``, cost must equal the plain
    ``input_tokens × input_price + output_tokens × output_price`` formula
    (no discount applied).
    """
    mock_require_dep.return_value = _make_fake_genai_module()

    with settings.edit(persist=False):
        settings.GOOGLE_API_KEY = "test-key"

    model = GeminiModel(model="gemini-2.5-flash")
    reg = GEMINI_MODELS_DATA.get("gemini-2.5-flash")

    result_no_cache = model.calculate_cost(1000, 200, cached_tokens=0)
    result_baseline = model.calculate_cost(1000, 200)

    assert isinstance(result_no_cache, EvaluationCost)
    assert float(result_no_cache) == float(result_baseline)


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_token_cost_reads_cached_content_token_count(
    mock_require_dep, settings
):
    """
    ``_token_cost`` must read ``usage_metadata.cached_content_token_count``
    and pass it to ``calculate_cost``, producing a lower cost than a response
    with identical total tokens but no caching.
    """
    fake_genai = _make_fake_genai_module()
    fake_genai.types.GenerateContentConfig = lambda **kwargs: kwargs
    fake_client = MagicMock()

    cached_response = SimpleNamespace(
        text="hi",
        usage_metadata=SimpleNamespace(
            prompt_token_count=1000,
            candidates_token_count=200,
            cached_content_token_count=800,
        ),
    )
    uncached_response = SimpleNamespace(
        text="hi",
        usage_metadata=SimpleNamespace(
            prompt_token_count=1000,
            candidates_token_count=200,
        ),
    )

    def _fake_require_dependency(name, *args, **kwargs):
        if name == "google.genai":
            return fake_genai
        raise AssertionError(f"Unexpected dependency: {name}")

    mock_require_dep.side_effect = _fake_require_dependency

    with settings.edit(persist=False):
        settings.GOOGLE_API_KEY = "test-key"

    model = GeminiModel(model="gemini-2.5-flash")
    model.load_model = lambda *a, **kw: fake_client

    fake_client.models.generate_content.return_value = cached_response
    _, cost_cached = model.generate("prompt")

    fake_client.models.generate_content.return_value = uncached_response
    _, cost_uncached = model.generate("prompt")

    assert isinstance(cost_cached, EvaluationCost)
    assert float(cost_cached) < float(
        cost_uncached
    ), "context-cache discount must produce lower cost than full-price input"


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_no_cache_price_falls_back_to_full_input_price(
    mock_require_dep, settings
):
    """
    If a model has no ``cache_read_input_price``, even when ``cached_tokens``
    is non-zero the cost must still equal the plain full-price formula.
    """
    mock_require_dep.return_value = _make_fake_genai_module()

    with settings.edit(persist=False):
        settings.GOOGLE_API_KEY = "test-key"

    # Use a model that has no cache price (e.g. gemini-pro)
    model = GeminiModel(model="gemini-pro")
    model.model_data.cache_read_input_price = None
    # Give it pricing so calculate_cost doesn't return None
    model.model_data.input_price = 0.5 / 1e6
    model.model_data.output_price = 1.5 / 1e6

    result = model.calculate_cost(1000, 200, cached_tokens=800)
    expected = (
        1000 * model.model_data.input_price
        + 200 * model.model_data.output_price
    )

    assert isinstance(result, EvaluationCost)
    assert abs(float(result) - expected) < 1e-15


import pytest


@pytest.mark.parametrize(
    "model_name,expected_cache_price_approx",
    [
        ("gemini-1.5-pro", 0.3125e-6),
        ("gemini-1.5-pro-002", 0.3125e-6),
        ("gemini-1.5-flash", 0.01875e-6),
        ("gemini-1.5-flash-002", 0.01875e-6),
        ("gemini-1.5-flash-8b", 0.01e-6),
        ("gemini-2.0-flash", 0.0375e-6),
        ("gemini-2.5-pro", 0.3125e-6),
        ("gemini-2.5-flash", 0.0375e-6),
        ("gemini-2.5-flash-lite", 0.01875e-6),
    ],
)
def test_gemini_models_have_cache_read_input_price(
    model_name, expected_cache_price_approx
):
    """
    Every Gemini model that supports context caching must have a non-None
    ``cache_read_input_price`` in the registry at the expected rate.
    """
    data = GEMINI_MODELS_DATA.get(model_name)
    assert data is not None, f"{model_name} not in registry"
    assert (
        data.cache_read_input_price is not None
    ), f"{model_name} is missing cache_read_input_price"
    assert (
        abs(data.cache_read_input_price - expected_cache_price_approx) < 1e-12
    ), (
        f"{model_name}: expected {expected_cache_price_approx}, "
        f"got {data.cache_read_input_price}"
    )


@patch("deepeval.models.llms.gemini_model.require_dependency")
def test_gemini_cache_discount_cheaper_than_uncached_at_scale(
    mock_require_dep, settings
):
    """
    Sanity check: with 90% of tokens cached, total cost must be strictly
    cheaper than the no-cache baseline. The exact savings depend on the
    input/output token mix; cached tokens cost 25% of the full input price,
    so a token mix dominated by input shows larger savings than one dominated
    by output (output tokens are not discounted).
    """
    mock_require_dep.return_value = _make_fake_genai_module()

    with settings.edit(persist=False):
        settings.GOOGLE_API_KEY = "test-key"

    model = GeminiModel(model="gemini-2.5-pro")

    total_input = 100_000
    cached = 90_000  # 90% cached
    output = 5_000

    cost_with_cache = model.calculate_cost(total_input, output, cached)
    cost_no_cache = model.calculate_cost(total_input, output, 0)

    # gemini-2.5-pro: input $1.25/M, cache $0.3125/M (25%), output $10/M.
    # Savings on input = (1.25 - 0.3125) * 90K = $0.084.
    # cost_no_cache ≈ $0.175; cost_with_cache ≈ $0.091 (~52% of no-cache).
    # Assert cost is at least 40% cheaper (ratio < 0.6).
    assert (
        float(cost_with_cache) < float(cost_no_cache) * 0.6
    ), "With 90% cached, total cost should be at least 40% cheaper"
