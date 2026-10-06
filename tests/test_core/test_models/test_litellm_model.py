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


def test_litellm_model_name_resolves_provider_with_configured_region(
    monkeypatch, settings
):
    calls = {}

    def fake_get_llm_provider(model, api_base=None, litellm_params=None):
        calls["model"] = model
        calls["api_base"] = api_base
        calls["litellm_params"] = litellm_params
        return "resolved-provider"

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        types.SimpleNamespace(get_llm_provider=fake_get_llm_provider),
    )
    monkeypatch.setitem(
        sys.modules,
        "litellm.types.router",
        types.SimpleNamespace(GenericLiteLLMParams=dict),
    )

    model = LiteLLMModel(
        model="bedrock_mantle/openai.gpt-oss-120b",
        api_key="test-key",
        aws_region_name="us-east-1",
    )

    assert (
        model.get_model_name()
        == "bedrock_mantle/openai.gpt-oss-120b (resolved-provider)"
    )
    assert calls["model"] == "bedrock_mantle/openai.gpt-oss-120b"
    assert calls["api_base"] is None
    assert calls["litellm_params"] == {"aws_region_name": "us-east-1"}


##############################
# temperature fallback tests #
##############################


class _FakeUnsupportedParamsError(Exception):
    pass


def _patch_completion(monkeypatch, reject_temperature=True):
    """Make `litellm.completion`/`acompletion` reject any `temperature` the
    way LiteLLM does client-side for models that only accept the default,
    and record the params of every call."""
    litellm_module = sys.modules["litellm"]
    calls = []

    def fake_completion(**params):
        calls.append(params)
        if reject_temperature and "temperature" in params:
            raise _FakeUnsupportedParamsError(
                "o3 does not support temperature=0.0. Only temperature=1 is "
                "supported."
            )
        message = SimpleNamespace(content="OK")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=None
        )

    async def fake_acompletion(**params):
        return fake_completion(**params)

    monkeypatch.setattr(litellm_module, "completion", fake_completion)
    monkeypatch.setattr(litellm_module, "acompletion", fake_acompletion)
    monkeypatch.setattr(
        litellm_module,
        "UnsupportedParamsError",
        _FakeUnsupportedParamsError,
        raising=False,
    )
    return calls


def test_litellm_drops_default_temperature_when_model_rejects_it(
    monkeypatch, settings
):
    calls = _patch_completion(monkeypatch)
    model = _mk_litellm_model(settings)

    assert model.generate("hi") == ("OK", None)
    assert calls[0]["temperature"] == 0.0
    assert "temperature" not in calls[1]

    # Remembered for later calls: no second rejected request.
    model.generate("hi")
    assert len(calls) == 3
    assert "temperature" not in calls[2]


async def test_litellm_a_generate_drops_default_temperature_when_model_rejects_it(
    monkeypatch, settings
):
    calls = _patch_completion(monkeypatch)
    model = _mk_litellm_model(settings)

    assert await model.a_generate("hi") == ("OK", None)
    assert [("temperature" in c) for c in calls] == [True, False]


def test_litellm_keeps_explicit_temperature_when_model_rejects_it(
    monkeypatch, settings
):
    calls = _patch_completion(monkeypatch)
    model = _mk_litellm_model(settings, temperature=0)

    with pytest.raises(_FakeUnsupportedParamsError):
        model.generate("hi")
    assert len(calls) == 1


def test_litellm_sends_default_temperature_when_model_accepts_it(
    monkeypatch, settings
):
    calls = _patch_completion(monkeypatch, reject_temperature=False)
    model = _mk_litellm_model(settings)

    model.generate("hi")
    assert len(calls) == 1
    assert calls[0]["temperature"] == 0.0
