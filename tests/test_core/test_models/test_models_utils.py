import pytest
import logging
from pydantic import SecretStr

from deepeval.errors import DeepEvalError
from deepeval.models.utils import (
    require_secret_api_key,
    normalize_kwargs_and_extract_aliases,
)


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


def test_normalize_kwargs_and_extract_aliases_moves_aliases_and_logs(caplog):
    alias_map = {
        "model_name": ["model"],
        "api_key": ["_openai_api_key"],
    }
    original_kwargs = {
        "model": "gpt-4o",
        "_openai_api_key": "secret-key",
        "timeout": 30,
    }

    with caplog.at_level(logging.WARNING):
        normalized, extracted = normalize_kwargs_and_extract_aliases(
            "GPTModel",
            original_kwargs,
            alias_map,
        )

    # original kwargs should not be mutated
    assert original_kwargs == {
        "model": "gpt-4o",
        "_openai_api_key": "secret-key",
        "timeout": 30,
    }

    # legacy keys removed from normalized; canonical values returned via extracted
    assert normalized == {"timeout": 30}
    assert extracted == {
        "model_name": "gpt-4o",
        "api_key": "secret-key",
    }

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "keyword 'model' is deprecated" in messages
    assert "keyword '_openai_api_key' is deprecated" in messages


def test_normalize_kwargs_and_extract_aliases_no_alias_usage_no_logs(caplog):
    alias_map = {
        "model_name": ["model"],
    }
    kwargs = {"timeout": 30}

    with caplog.at_level(logging.WARNING):
        normalized, extracted = normalize_kwargs_and_extract_aliases(
            "GPTModel",
            kwargs,
            alias_map,
        )

    # nothing changed
    assert normalized == {"timeout": 30}
    assert extracted == {}

    # no warnings logged
    assert caplog.records == []
