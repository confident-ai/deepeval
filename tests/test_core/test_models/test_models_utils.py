import pytest
from pydantic import SecretStr

from deepeval.errors import DeepEvalError
from deepeval.models.utils import require_secret_api_key


def test_require_secret_api_key_raises_when_none():
    with pytest.raises(DeepEvalError, match="not configured"):
        require_secret_api_key(
            None,
            provider_label="Anthropic",
            env_var_name="ANTHROPIC_API_KEY",
            param_hint="`_anthropic_api_key` to AnthropicModel(...)",
        )


def test_require_secret_api_key_raises_when_empty():
    with pytest.raises(DeepEvalError, match="empty"):
        require_secret_api_key(
            SecretStr(""),
            provider_label="Anthropic",
            env_var_name="ANTHROPIC_API_KEY",
            param_hint="`_anthropic_api_key` to AnthropicModel(...)",
        )


def test_require_secret_api_key_returns_plain_string_for_valid_secret():
    secret = SecretStr("real-key")
    result = require_secret_api_key(
        secret,
        provider_label="Anthropic",
        env_var_name="ANTHROPIC_API_KEY",
        param_hint="`_anthropic_api_key` to AnthropicModel(...)",
    )
    assert result == "real-key"
    assert isinstance(result, str)
