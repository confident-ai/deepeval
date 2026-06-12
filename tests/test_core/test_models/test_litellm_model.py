import sys
import types
import pytest
from types import SimpleNamespace
from pydantic import SecretStr

from deepeval.errors import DeepEvalError
from deepeval.models.llms.litellm_model import LiteLLMModel  # noqa: E402

############################################################################
# Stub a fake `litellm` module so LiteLLMModel can import it even when the #
# real dependency is not installed.                                        #
############################################################################


if "litellm" not in sys.modules:
    fake_litellm = types.SimpleNamespace(
        completion=lambda *a, **k: None,
        acompletion=lambda *a, **k: None,
        get_llm_provider=lambda model: "stub-provider",
    )
    sys.modules["litellm"] = fake_litellm


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
# cost unit tests            #
##############################
#
# LiteLLM is now standardized on the shared gateway cost contract:
#   1. user-supplied per-token pricing,
#   2. a cost reported by the gateway on the response, otherwise
#   3. None (no more inventing hardcoded per-token rates).


def _mk_litellm_model(settings, **kwargs):
    with settings.edit(persist=False):
        settings.LITELLM_MODEL_NAME = "test-model"
        settings.LITELLM_API_KEY = "test-key"
    return LiteLLMModel(**kwargs)


def _mk_response(prompt_tokens=100, completion_tokens=50, cost=None):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    resp = SimpleNamespace(usage=usage)
    if cost is not None:
        resp.cost = cost
    return resp


def test_litellm_cost_prefers_reported_response_cost(settings):
    model = _mk_litellm_model(settings)
    response = _mk_response(prompt_tokens=100, completion_tokens=50, cost=0.042)
    cost = model._response_cost(response)
    assert cost == 0.042
    assert cost.input_tokens == 100
    assert cost.output_tokens == 50


def test_litellm_cost_is_none_when_unknown(settings):
    # No user pricing and no cost reported by the gateway -> unknown.
    model = _mk_litellm_model(settings)
    response = _mk_response(prompt_tokens=200, completion_tokens=100, cost=None)
    assert model._response_cost(response) is None


def test_litellm_cost_uses_user_pricing(settings):
    model = _mk_litellm_model(
        settings, cost_per_input_token=0.0001, cost_per_output_token=0.0002
    )
    response = _mk_response(prompt_tokens=100, completion_tokens=50)
    cost = model._response_cost(response)
    assert cost == pytest.approx((100 * 0.0001) + (50 * 0.0002))


def test_litellm_cost_handles_missing_usage_gracefully(settings):
    model = _mk_litellm_model(settings)
    assert model._response_cost(SimpleNamespace()) is None
