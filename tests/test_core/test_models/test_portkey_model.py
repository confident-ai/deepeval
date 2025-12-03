import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import SecretStr

from deepeval.errors import DeepEvalError
from deepeval.models.llms.portkey_model import PortkeyModel


def make_settings(**overrides):
    """Return a dummy settings object with PORTKEY defaults."""
    defaults = dict(
        PORTKEY_MODEL_NAME="gpt-4o-mini",
        PORTKEY_API_KEY=SecretStr("portkey-secret"),
        PORTKEY_BASE_URL="https://api.portkey.ai/v1",
        PORTKEY_PROVIDER_NAME="openai",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


#####################################
# __init__ / configuration behavior #
#####################################


@patch("deepeval.models.llms.portkey_model.get_settings")
def test_portkey_model_prefers_explicit_params_over_settings(mock_get_settings):
    mock_get_settings.return_value = make_settings(
        PORTKEY_MODEL_NAME="from-settings-model",
        PORTKEY_BASE_URL="https://from-settings.example.com",
        PORTKEY_PROVIDER_NAME="from-settings-provider",
        PORTKEY_API_KEY=SecretStr("settings-secret"),
    )

    model = PortkeyModel(
        model="explicit-model",
        api_key="explicit-secret",
        base_url="https://explicit.example.com/",
        provider="explicit-provider",
    )

    # Explicit params should win over settings
    assert model.model == "explicit-model"
    assert (
        model.base_url == "https://explicit.example.com"
    )  # trailing slash stripped
    assert model.provider == "explicit-provider"

    # _headers should use the explicit api_key
    headers = model._headers()
    assert headers["x-portkey-api-key"] == "explicit-secret"
    assert headers["x-portkey-provider"] == "explicit-provider"


@patch("deepeval.models.llms.portkey_model.get_settings")
def test_portkey_model_uses_settings_when_params_missing(mock_get_settings):
    mock_get_settings.return_value = make_settings()

    model = PortkeyModel()

    assert model.model == "gpt-4o-mini"
    assert model.base_url == "https://api.portkey.ai/v1"
    assert model.provider == "openai"

    headers = model._headers()
    # SecretStr should be unwrapped by require_secret_api_key
    assert headers["x-portkey-api-key"] == "portkey-secret"
    assert headers["x-portkey-provider"] == "openai"


@patch("deepeval.models.llms.portkey_model.get_settings")
def test_portkey_model_raises_if_model_missing(mock_get_settings):
    # Model missing both as arg and in settings
    mock_get_settings.return_value = make_settings(
        PORTKEY_MODEL_NAME=None,
    )

    with pytest.raises(DeepEvalError) as exc:
        PortkeyModel(model=None)

    msg = str(exc.value)
    assert "Portkey is missing a required parameter" in msg
    assert "PORTKEY_MODEL_NAME" in msg
    assert "model" in msg


@patch("deepeval.models.llms.portkey_model.get_settings")
def test_portkey_model_raises_if_base_url_missing(mock_get_settings):
    # Model present but base URL missing in both places
    mock_get_settings.return_value = make_settings(
        PORTKEY_BASE_URL=None,
    )

    with pytest.raises(DeepEvalError) as exc:
        PortkeyModel(model="gpt-4o-mini", base_url=None)

    msg = str(exc.value)
    assert "Portkey is missing a required parameter" in msg
    assert "PORTKEY_BASE_URL" in msg
    assert "base_url" in msg


@patch("deepeval.models.llms.portkey_model.get_settings")
def test_portkey_model_raises_if_provider_missing(mock_get_settings):
    # Model and base URL present, provider missing
    mock_get_settings.return_value = make_settings(
        PORTKEY_PROVIDER_NAME=None,
    )

    with pytest.raises(DeepEvalError) as exc:
        PortkeyModel(model="gpt-4o-mini", base_url="https://api.portkey.ai/v1")

    msg = str(exc.value)
    assert "Portkey is missing a required parameter" in msg
    assert "PORTKEY_PROVIDER_NAME" in msg
    assert "provider" in msg


##############
# generate() #
##############


@patch("deepeval.models.llms.portkey_model.requests.post")
@patch("deepeval.models.llms.portkey_model.get_settings")
def test_portkey_generate_sends_request_and_returns_content(
    mock_get_settings, mock_post
):
    mock_get_settings.return_value = make_settings()

    model = PortkeyModel()
    prompt = "Hello from DeepEval"

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Portkey says hi!",
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    output = model.generate(prompt)

    assert output == "Portkey says hi!"
    mock_post.assert_called_once()

    args, kwargs = mock_post.call_args
    # URL
    assert args[0] == f"{model.base_url}/chat/completions"
    # Payload
    assert kwargs["json"] == model._payload(prompt)
    # Headers
    headers = kwargs["headers"]
    assert headers["x-portkey-api-key"] == "portkey-secret"
    assert headers["x-portkey-provider"] == "openai"
    assert headers["Content-Type"] == "application/json"


################
# a_generate() #
################


@pytest.mark.asyncio
@patch("deepeval.models.llms.portkey_model.get_settings")
async def test_portkey_a_generate_sends_request_and_returns_content(
    mock_get_settings,
):
    mock_get_settings.return_value = make_settings()

    model = PortkeyModel()
    prompt = "Hello from async DeepEval"

    # Mock the response object returned inside the inner async with
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Portkey async hi!",
                }
            }
        ]
    }

    # Context manager returned by call to session.post
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

    # session object from aiohttp.ClientSession
    mock_session = MagicMock()
    # async with ClientSession() as session -> session is mock_session
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    # call to session.post should return our post context manager
    mock_session.post = MagicMock(return_value=mock_post_ctx)

    # Patch ClientSession() to return mock_session
    with patch(
        "deepeval.models.llms.portkey_model.aiohttp.ClientSession",
        return_value=mock_session,
    ):
        output = await model.a_generate(prompt)

    assert output == "Portkey async hi!"

    # Verify we called the right URL with the right payload & headers
    mock_session.post.assert_called_once()
    args, kwargs = mock_session.post.call_args
    assert args[0] == f"{model.base_url}/chat/completions"
    assert kwargs["json"] == model._payload(prompt)
    headers = kwargs["headers"]
    assert headers["x-portkey-api-key"] == "portkey-secret"
    assert headers["x-portkey-provider"] == "openai"
    assert headers["Content-Type"] == "application/json"
