import math
import sys
import types
import pytest
from types import SimpleNamespace
from pydantic import SecretStr

from deepeval.errors import DeepEvalError
from deepeval.models.utils import EvaluationCost

# Stub `litellm` so LiteLLMModel imports cleanly without the real package.
if "litellm" not in sys.modules:
    sys.modules["litellm"] = types.SimpleNamespace(
        completion=lambda *a, **k: None,
        acompletion=lambda *a, **k: None,
        get_llm_provider=lambda model: "stub-provider",
        completion_cost=lambda **k: None,
    )
elif isinstance(sys.modules["litellm"], types.SimpleNamespace) and not hasattr(
    sys.modules["litellm"], "completion_cost"
):
    # Only ever patch our own test stub — never the real litellm module.
    sys.modules["litellm"].completion_cost = lambda **k: None

from deepeval.models.llms.litellm_model import LiteLLMModel  # noqa: E402


def test_litellm_explicit_overrides_settings_and_env(monkeypatch, settings):
    """
    Explicit ctor `model`, `api_key`, and `api_base` must override both
    Settings-derived defaults and any environment variables.
    """

    # Seed env vars that are part of the fallback chain, but must be ignored
    # when ctor args are explicitly provided.
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "env-proxy-key")
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "env-google-key")
    monkeypatch.setenv("LITELLM_API_BASE", "http://env-base")
    monkeypatch.setenv("LITELLM_PROXY_API_BASE", "http://env-proxy-base")

    # Seed Settings with defaults that should not be used when ctor
    # arguments are provided.
    with settings.edit(persist=False):
        settings.LITELLM_MODEL_NAME = "settings-model"
        settings.LITELLM_API_KEY = "settings-api-key"
        settings.LITELLM_API_BASE = "http://settings-base"

    # Explicit ctor values must win over both Settings and environment
    model = LiteLLMModel(
        model="ctor-model",
        api_key="ctor-api-key",
        base_url="http://ctor-base",
    )

    # Model name and connection parameters should come from ctor arguments
    assert model.name == "ctor-model"
    assert isinstance(model.api_key, SecretStr)
    assert model.api_key.get_secret_value() == "ctor-api-key"
    assert model.base_url is not None
    assert model.base_url.rstrip("/") == "http://ctor-base"


def test_litellm_defaults_model_api_key_and_base_from_settings(settings):
    """
    When no ctor `model`, `api_key`, or `api_base` are provided, LiteLLMModel
    should resolve all three from the Pydantic Settings object:

      - model from Settings.LITELLM_MODEL_NAME
      - api_key    from Settings.LITELLM_API_KEY
      - api_base   from Settings.LITELLM_API_BASE
    """

    # Seed Settings with the values that should be used by default
    with settings.edit(persist=False):
        settings.LITELLM_MODEL_NAME = "settings-model"
        settings.LITELLM_API_KEY = "settings-api-key"
        settings.LITELLM_API_BASE = "http://settings-base"

    # No ctor overrides: values must be resolved from Settings
    model = LiteLLMModel()

    assert model.name == "settings-model"
    assert isinstance(model.api_key, SecretStr)
    assert model.api_key.get_secret_value() == "settings-api-key"
    assert model.base_url is not None
    assert model.base_url.rstrip("/") == "http://settings-base"


def test_litellm_raises_when_model_missing(settings):
    """
    If neither ctor `model` nor Settings.LITELLM_MODEL_NAME is set,
    LiteLLMModel should raise a DeepEvalError.
    """
    # Clear any model name in Settings
    with settings.edit(persist=False):
        settings.LITELLM_MODEL_NAME = None

    with pytest.raises(DeepEvalError):
        LiteLLMModel()


########################################################
# Test legacy keyword backwards compatability behavior #
########################################################


def test_litellm_model_accepts_legacy_api_base_keyword_and_maps_to_base_url(
    settings,
):
    with settings.edit(persist=False):
        settings.LITELLM_MODEL_NAME = "settings-model"
        settings.LITELLM_API_KEY = "settings-api-key"

    model = LiteLLMModel(base_url="http://ctor-base")

    # legacy keyword mapped to canonical parameter
    assert model.base_url == "http://ctor-base"

    # legacy key should not be forwarded to the client kwargs
    assert "api_base" not in model.kwargs


##############################
# calculate_cost unit tests  #
##############################


def _mk_litellm_model(settings, model_name="test-model"):
    with settings.edit(persist=False):
        settings.LITELLM_MODEL_NAME = model_name
        settings.LITELLM_API_KEY = "test-key"
    return LiteLLMModel()


def _mk_response(prompt_tokens=100, completion_tokens=50, cost=None):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    resp = SimpleNamespace(usage=usage)
    if cost is not None:
        resp.cost = cost
    return resp


@pytest.fixture
def patch_completion_cost(monkeypatch):
    """Swap ``litellm.completion_cost`` for the duration of one test."""

    def _set(fn):
        monkeypatch.setattr(sys.modules["litellm"], "completion_cost", fn)

    yield _set


def test_litellm_calculate_cost_prefers_explicit_response_cost(
    settings, patch_completion_cost
):
    """An explicit ``response.cost`` wins; the registry isn't even consulted."""
    called = {"hit": False}

    def boom(**_):
        called["hit"] = True
        return 999.0

    patch_completion_cost(boom)

    model = _mk_litellm_model(settings)
    response = _mk_response(prompt_tokens=100, completion_tokens=50, cost=0.042)
    cost = model.calculate_cost(response)

    assert isinstance(cost, EvaluationCost)
    assert cost == 0.042
    assert cost.input_tokens == 100
    assert cost.output_tokens == 50
    assert called["hit"] is False


def test_litellm_calculate_cost_uses_litellm_completion_cost(
    settings, patch_completion_cost
):
    """Without an explicit cost, fall through to LiteLLM's pricing API."""
    patch_completion_cost(lambda **_: 0.000275)

    model = _mk_litellm_model(settings)
    response = _mk_response(prompt_tokens=1000, completion_tokens=500)
    cost = model.calculate_cost(response)

    assert isinstance(cost, EvaluationCost)
    assert cost == pytest.approx(0.000275)
    assert cost.input_tokens == 1000
    assert cost.output_tokens == 500


def test_litellm_calculate_cost_does_not_fabricate_per_token_rates(
    settings, patch_completion_cost
):
    """Regression guard: the old fallback charged $0.0001/$0.0002 per token —
    7–666× over real prices. Unknown pricing must report 0.0, not invent one."""
    patch_completion_cost(lambda **_: None)

    model = _mk_litellm_model(settings, model_name="some-unregistered-model")
    response = _mk_response(prompt_tokens=100, completion_tokens=50)
    cost = model.calculate_cost(response)

    fabricated = (100 * 0.0001) + (50 * 0.0002)
    assert cost != fabricated
    assert cost == 0.0
    assert cost.input_tokens == 100
    assert cost.output_tokens == 50


def test_litellm_calculate_cost_returns_zero_when_completion_cost_raises(
    settings, patch_completion_cost, caplog
):
    """If LiteLLM raises (unknown model), report 0.0 + log — never crash."""

    def raises(**_):
        raise Exception("Model not in pricing registry: foo/bar")

    patch_completion_cost(raises)

    model = _mk_litellm_model(settings, model_name="foo/bar")
    response = _mk_response(prompt_tokens=200, completion_tokens=100)

    with caplog.at_level("WARNING"):
        cost = model.calculate_cost(response)

    assert cost == 0.0
    assert cost.input_tokens == 200
    assert cost.output_tokens == 100
    assert any(
        "LiteLLM cost lookup failed" in rec.message for rec in caplog.records
    )


def test_litellm_calculate_cost_accumulates_evaluation_cost(
    settings, patch_completion_cost
):
    """Per-call costs add up cleanly into ``evaluation_cost``."""
    patch_completion_cost(lambda **_: None)

    model = _mk_litellm_model(settings)
    assert model.evaluation_cost == 0.0

    model.calculate_cost(_mk_response(cost=0.01))
    model.calculate_cost(_mk_response(cost=0.02))
    model.calculate_cost(_mk_response(cost=0.03))

    assert model.evaluation_cost == pytest.approx(0.06)
    assert model.get_evaluation_cost() == pytest.approx(0.06)


def test_litellm_calculate_cost_accumulator_unaffected_by_unknown_price(
    settings, patch_completion_cost
):
    """Unknown-price calls add 0 — totals stay honest across mixed runs."""
    patch_completion_cost(lambda **_: None)

    model = _mk_litellm_model(settings)
    model.calculate_cost(_mk_response(cost=0.10))
    model.calculate_cost(_mk_response(prompt_tokens=500, completion_tokens=500))
    model.calculate_cost(_mk_response(cost=0.05))

    assert model.evaluation_cost == pytest.approx(0.15)


def test_litellm_calculate_cost_with_zero_tokens_no_response_cost(
    settings, patch_completion_cost
):
    """Zero tokens, no price → zero cost, accumulator untouched."""
    patch_completion_cost(lambda **_: None)

    model = _mk_litellm_model(settings)
    response = _mk_response(prompt_tokens=0, completion_tokens=0)
    cost = model.calculate_cost(response)

    assert cost == 0.0
    assert model.evaluation_cost == 0.0


def test_litellm_calculate_cost_handles_malformed_response_gracefully(
    settings, patch_completion_cost
):
    """A response missing ``.usage`` must not crash the metric run."""
    patch_completion_cost(lambda **_: None)
    model = _mk_litellm_model(settings)
    bad_response = SimpleNamespace()
    cost = model.calculate_cost(bad_response)
    assert cost == 0.0
    assert cost.input_tokens is None
    assert cost.output_tokens is None


def test_litellm_calculate_cost_handles_non_numeric_explicit_cost(
    settings, patch_completion_cost, caplog
):
    """A junk ``response.cost`` (bad proxy payload) falls through to
    completion_cost AND emits a warning so operators see the broken proxy."""
    patch_completion_cost(lambda **_: 0.007)

    model = _mk_litellm_model(settings)
    response = _mk_response(prompt_tokens=10, completion_tokens=5)
    response.cost = "not-a-number"

    with caplog.at_level("WARNING"):
        cost = model.calculate_cost(response)

    assert cost == pytest.approx(0.007)
    assert cost.input_tokens == 10
    assert cost.output_tokens == 5
    assert any(
        "ignoring invalid response.cost" in rec.message
        for rec in caplog.records
    )


###################################################
# Hardening: NaN / Inf / negative / bool / dict   #
###################################################


def test_litellm_calculate_cost_rejects_nan(settings, patch_completion_cost):
    """NaN in response.cost would poison the accumulator forever (NaN + x
    == NaN). Must be treated as invalid and report 0.0."""
    patch_completion_cost(lambda **_: None)
    model = _mk_litellm_model(settings)
    response = _mk_response(
        prompt_tokens=10, completion_tokens=5, cost=float("nan")
    )
    cost = model.calculate_cost(response)

    assert cost == 0.0
    assert model.evaluation_cost == 0.0
    assert not math.isnan(model.evaluation_cost)
    assert cost.input_tokens == 10
    assert cost.output_tokens == 5


def test_litellm_calculate_cost_rejects_infinity(
    settings, patch_completion_cost
):
    """+inf would also poison the accumulator."""
    patch_completion_cost(lambda **_: None)
    model = _mk_litellm_model(settings)
    response = _mk_response(cost=float("inf"))
    cost = model.calculate_cost(response)

    assert cost == 0.0
    assert model.evaluation_cost == 0.0


def test_litellm_calculate_cost_rejects_negative(
    settings, patch_completion_cost
):
    """A negative cost (proxy bug) must not subtract from the accumulator."""
    patch_completion_cost(lambda **_: None)
    model = _mk_litellm_model(settings)
    response = _mk_response(cost=-1.5)
    cost = model.calculate_cost(response)

    assert cost == 0.0
    assert model.evaluation_cost == 0.0


def test_litellm_calculate_cost_rejects_bool(settings, patch_completion_cost):
    """bool is an int subclass — float(True) == 1.0 would silently bill $1
    per call. Reject explicitly."""
    patch_completion_cost(lambda **_: None)
    model = _mk_litellm_model(settings)
    response = _mk_response(cost=True)
    cost = model.calculate_cost(response)

    assert cost == 0.0
    assert model.evaluation_cost == 0.0


def test_litellm_calculate_cost_reads_dict_usage(
    settings, patch_completion_cost
):
    """Some LiteLLM proxy / raw-JSON responses serialize ``usage`` as a
    plain dict. ``getattr`` does not read dict keys, so the previous code
    silently dropped token counts."""
    patch_completion_cost(lambda **_: 0.005)
    model = _mk_litellm_model(settings)
    response = SimpleNamespace(
        usage={"prompt_tokens": 100, "completion_tokens": 50},
    )
    cost = model.calculate_cost(response)

    assert cost == pytest.approx(0.005)
    assert cost.input_tokens == 100
    assert cost.output_tokens == 50


def test_litellm_calculate_cost_falls_back_to_usage_cost(
    settings, patch_completion_cost
):
    """When ``response.cost`` is unset but ``response.usage.cost`` is set
    (OpenRouter-style), use it and skip the pricing registry."""
    called = {"hit": False}

    def boom(**_):
        called["hit"] = True
        return 0.0

    patch_completion_cost(boom)

    model = _mk_litellm_model(settings)
    usage = SimpleNamespace(
        prompt_tokens=100, completion_tokens=50, cost=0.0123
    )
    response = SimpleNamespace(usage=usage)
    cost = model.calculate_cost(response)

    assert cost == pytest.approx(0.0123)
    assert cost.input_tokens == 100
    assert cost.output_tokens == 50
    assert called["hit"] is False


def test_litellm_calculate_cost_handles_completion_cost_import_error(
    settings, monkeypatch, caplog
):
    """If ``from litellm import completion_cost`` fails (old install, broken
    package), the warning must clearly point at the missing symbol — not
    a generic 'lookup failed'."""
    original = sys.modules["litellm"]
    monkeypatch.setitem(
        sys.modules,
        "litellm",
        types.SimpleNamespace(
            completion=original.completion,
            acompletion=original.acompletion,
            get_llm_provider=original.get_llm_provider,
            # completion_cost intentionally absent.
        ),
    )

    model = _mk_litellm_model(settings)
    response = _mk_response(prompt_tokens=10, completion_tokens=5)

    with caplog.at_level("WARNING"):
        cost = model.calculate_cost(response)

    assert cost == 0.0
    assert any(
        "litellm.completion_cost unavailable" in rec.message
        for rec in caplog.records
    )


def test_litellm_calculate_cost_rejects_invalid_completion_cost_value(
    settings, patch_completion_cost, caplog
):
    """If litellm.completion_cost itself returns NaN/negative (its own bug),
    we must not let it poison our accumulator."""
    patch_completion_cost(lambda **_: float("nan"))
    model = _mk_litellm_model(settings)
    response = _mk_response(prompt_tokens=10, completion_tokens=5)

    with caplog.at_level("WARNING"):
        cost = model.calculate_cost(response)

    assert cost == 0.0
    assert model.evaluation_cost == 0.0
    assert any(
        "litellm.completion_cost returned invalid value" in rec.message
        for rec in caplog.records
    )


###############################################
# Raw-response paths preserve EvaluationCost  #
###############################################


def test_litellm_generate_raw_response_preserves_evaluation_cost_subclass(
    settings, patch_completion_cost, monkeypatch
):
    """``generate_raw_response`` previously wrapped its cost in ``float()``,
    stripping the ``EvaluationCost`` subclass — so ``accrue_token_usage``'s
    isinstance check silently dropped tokens for g_eval / log-probs paths."""
    patch_completion_cost(lambda **_: 0.001)

    fake_response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=42, completion_tokens=7)
    )
    monkeypatch.setattr(
        sys.modules["litellm"], "completion", lambda **_: fake_response
    )

    model = _mk_litellm_model(settings)
    _resp, cost = model.generate_raw_response("prompt")

    assert isinstance(cost, EvaluationCost)
    assert cost.input_tokens == 42
    assert cost.output_tokens == 7
