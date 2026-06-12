import pytest
from unittest.mock import Mock, MagicMock, patch
from pydantic import BaseModel

from deepeval.errors import DeepEvalError
from deepeval.models.llms.portkey_model import PortkeyModel


class SampleSchema(BaseModel):
    field1: str
    field2: int


def _seed_settings(settings):
    with settings.edit(persist=False):
        settings.PORTKEY_MODEL_NAME = "gpt-4o-mini"
        settings.PORTKEY_BASE_URL = "https://api.portkey.ai/v1"
        settings.PORTKEY_PROVIDER_NAME = "openai"
        settings.PORTKEY_API_KEY = "portkey-secret"


#####################################
# __init__ / configuration behavior #
#####################################


def test_portkey_model_prefers_explicit_params_over_settings(settings):
    _seed_settings(settings)

    model = PortkeyModel(
        model="explicit-model",
        api_key="explicit-secret",
        base_url="https://explicit.example.com/",
        provider="explicit-provider",
    )

    # Explicit params should win over settings
    assert model.name == "explicit-model"
    assert model.base_url == "https://explicit.example.com"  # slash stripped
    assert model.provider == "explicit-provider"

    # Portkey auth headers are injected as OpenAI-client default headers
    headers = model._client_extra_kwargs()["default_headers"]
    assert headers["x-portkey-api-key"] == "explicit-secret"
    assert headers["x-portkey-provider"] == "explicit-provider"


def test_portkey_model_uses_settings_when_params_missing(settings):
    _seed_settings(settings)

    model = PortkeyModel()

    assert model.name == "gpt-4o-mini"
    assert model.base_url == "https://api.portkey.ai/v1"
    assert model.provider == "openai"

    headers = model._client_extra_kwargs()["default_headers"]
    # SecretStr should be unwrapped by require_secret_api_key
    assert headers["x-portkey-api-key"] == "portkey-secret"
    assert headers["x-portkey-provider"] == "openai"


def test_portkey_model_raises_if_model_missing(settings):
    _seed_settings(settings)
    with settings.edit(persist=False):
        settings.PORTKEY_MODEL_NAME = None

    with pytest.raises(DeepEvalError) as exc:
        PortkeyModel(model=None)

    msg = str(exc.value)
    assert "Portkey is missing a required parameter" in msg
    assert "PORTKEY_MODEL_NAME" in msg
    assert "model" in msg


def test_portkey_model_raises_if_base_url_missing(settings):
    _seed_settings(settings)
    with settings.edit(persist=False):
        settings.PORTKEY_BASE_URL = None

    with pytest.raises(DeepEvalError) as exc:
        PortkeyModel(model="gpt-4o-mini", base_url=None)

    msg = str(exc.value)
    assert "Portkey is missing a required parameter" in msg
    assert "PORTKEY_BASE_URL" in msg
    assert "base_url" in msg


def test_portkey_model_raises_if_provider_missing(settings):
    _seed_settings(settings)
    with settings.edit(persist=False):
        settings.PORTKEY_PROVIDER_NAME = None

    with pytest.raises(DeepEvalError) as exc:
        PortkeyModel(model="gpt-4o-mini", base_url="https://api.portkey.ai/v1")

    msg = str(exc.value)
    assert "Portkey is missing a required parameter" in msg
    assert "PORTKEY_PROVIDER_NAME" in msg
    assert "provider" in msg


def test_portkey_get_model_name(settings):
    _seed_settings(settings)
    model = PortkeyModel()
    assert model.get_model_name() == "gpt-4o-mini (Portkey)"


##############
# generate() #
##############


@patch("deepeval.models.llms.gateway_model.AsyncOpenAI")
def test_portkey_generate_returns_output_and_cost_tuple(
    mock_async_openai_class, settings
):
    _seed_settings(settings)

    mock_client = MagicMock()
    mock_async_openai_class.return_value = mock_client
    mock_completion = Mock()
    mock_completion.choices = [Mock(message=Mock(content="Portkey says hi!"))]
    mock_completion.usage.prompt_tokens = 10
    mock_completion.usage.completion_tokens = 20

    call_args = {}

    async def async_create(*args, **kwargs):
        call_args.update(kwargs)
        return mock_completion

    mock_client.chat.completions.create = async_create

    model = PortkeyModel()
    result = model.generate("Hello from DeepEval")

    # Standard gateway contract: (output, cost) tuple
    assert isinstance(result, tuple) and len(result) == 2
    output, _cost = result
    assert output == "Portkey says hi!"
    assert call_args["model"] == "gpt-4o-mini"
    assert call_args["messages"] == [
        {"role": "user", "content": "Hello from DeepEval"}
    ]


@patch("deepeval.models.llms.gateway_model.AsyncOpenAI")
async def test_portkey_a_generate_returns_output_and_cost_tuple(
    mock_async_openai_class, settings
):
    _seed_settings(settings)

    mock_client = MagicMock()
    mock_async_openai_class.return_value = mock_client
    mock_completion = Mock()
    mock_completion.choices = [Mock(message=Mock(content="Portkey async hi!"))]
    mock_completion.usage.prompt_tokens = 5
    mock_completion.usage.completion_tokens = 7

    async def async_create(*args, **kwargs):
        return mock_completion

    mock_client.chat.completions.create = async_create

    model = PortkeyModel()
    output, _cost = await model.a_generate("Hello from async DeepEval")

    assert output == "Portkey async hi!"


@patch("deepeval.models.llms.gateway_model.AsyncOpenAI")
def test_portkey_generate_with_structured_outputs(
    mock_async_openai_class, settings
):
    _seed_settings(settings)

    mock_client = MagicMock()
    mock_async_openai_class.return_value = mock_client
    mock_completion = Mock()
    mock_completion.choices = [
        Mock(message=Mock(content='{"field1": "test", "field2": 42}'))
    ]
    mock_completion.usage.prompt_tokens = 10
    mock_completion.usage.completion_tokens = 20

    call_args = {}

    async def async_create(*args, **kwargs):
        call_args.update(kwargs)
        return mock_completion

    mock_client.chat.completions.create = async_create

    model = PortkeyModel()
    output, _cost = model.generate("prompt", schema=SampleSchema)

    assert call_args["response_format"]["type"] == "json_schema"
    assert isinstance(output, SampleSchema)
    assert output.field1 == "test"
    assert output.field2 == 42


###################
# client building #
###################


@patch("deepeval.models.llms.gateway_model.OpenAI")
def test_portkey_build_client_sets_auth_headers(mock_openai_class, settings):
    _seed_settings(settings)

    model = PortkeyModel()
    _ = model.load_model(async_mode=False)

    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["base_url"] == "https://api.portkey.ai/v1"
    headers = call_kwargs["default_headers"]
    assert headers["x-portkey-api-key"] == "portkey-secret"
    assert headers["x-portkey-provider"] == "openai"


def test_portkey_is_recognized_as_native_model(settings):
    from deepeval.metrics.utils import initialize_model, is_native_model

    _seed_settings(settings)
    model = PortkeyModel()
    assert is_native_model(model)

    returned_model, using_native = initialize_model(model)
    assert using_native is True
    assert returned_model is model
