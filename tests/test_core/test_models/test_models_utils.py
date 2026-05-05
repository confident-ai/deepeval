import pytest
import logging
from pydantic import SecretStr

from deepeval.errors import DeepEvalError
from deepeval.models.base_model import DeepEvalModelData
from deepeval.models.utils import (
    require_secret_api_key,
    require_costs,
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


##############################
# require_costs unit tests   #
##############################


def test_require_costs_returns_registry_prices_when_both_present():
    model_data = DeepEvalModelData(input_price=0.01, output_price=0.02)
    inp, out = require_costs(
        model_data,
        model_name="test-model",
        input_token_envvar="TEST_COST_PER_INPUT_TOKEN",
        output_token_envvar="TEST_COST_PER_OUTPUT_TOKEN",
    )
    assert inp == 0.01
    assert out == 0.02


def test_require_costs_registry_prices_win_over_constructor_args():
    model_data = DeepEvalModelData(input_price=0.01, output_price=0.02)
    inp, out = require_costs(
        model_data,
        model_name="test-model",
        input_token_envvar="TEST_COST_PER_INPUT_TOKEN",
        output_token_envvar="TEST_COST_PER_OUTPUT_TOKEN",
        cost_per_input_token=0.99,
        cost_per_output_token=0.88,
    )
    assert inp == 0.01
    assert out == 0.02


def test_require_costs_uses_constructor_args_when_registry_missing():
    model_data = DeepEvalModelData(input_price=None, output_price=None)
    inp, out = require_costs(
        model_data,
        model_name="unknown-model",
        input_token_envvar="TEST_COST_PER_INPUT_TOKEN",
        output_token_envvar="TEST_COST_PER_OUTPUT_TOKEN",
        cost_per_input_token=0.05,
        cost_per_output_token=0.10,
    )
    assert inp == 0.05
    assert out == 0.10


def test_require_costs_returns_none_when_registry_and_constructor_missing():
    model_data = DeepEvalModelData(input_price=None, output_price=None)
    inp, out = require_costs(
        model_data,
        model_name="unknown-model",
        input_token_envvar="TEST_COST_PER_INPUT_TOKEN",
        output_token_envvar="TEST_COST_PER_OUTPUT_TOKEN",
    )
    assert inp is None
    assert out is None


def test_require_costs_returns_none_when_only_input_constructor_arg():
    model_data = DeepEvalModelData(input_price=None, output_price=None)
    inp, out = require_costs(
        model_data,
        model_name="unknown-model",
        input_token_envvar="TEST_COST_PER_INPUT_TOKEN",
        output_token_envvar="TEST_COST_PER_OUTPUT_TOKEN",
        cost_per_input_token=0.05,
    )
    assert inp is None
    assert out is None


def test_require_costs_returns_none_when_only_output_constructor_arg():
    model_data = DeepEvalModelData(input_price=None, output_price=None)
    inp, out = require_costs(
        model_data,
        model_name="unknown-model",
        input_token_envvar="TEST_COST_PER_INPUT_TOKEN",
        output_token_envvar="TEST_COST_PER_OUTPUT_TOKEN",
        cost_per_output_token=0.10,
    )
    assert inp is None
    assert out is None


def test_require_costs_raises_on_negative_input_cost():
    model_data = DeepEvalModelData(input_price=None, output_price=None)
    with pytest.raises(DeepEvalError, match="must be >= 0"):
        require_costs(
            model_data,
            model_name="test-model",
            input_token_envvar="TEST_COST_PER_INPUT_TOKEN",
            output_token_envvar="TEST_COST_PER_OUTPUT_TOKEN",
            cost_per_input_token=-0.01,
            cost_per_output_token=0.02,
        )


def test_require_costs_raises_on_negative_output_cost():
    model_data = DeepEvalModelData(input_price=None, output_price=None)
    with pytest.raises(DeepEvalError, match="must be >= 0"):
        require_costs(
            model_data,
            model_name="test-model",
            input_token_envvar="TEST_COST_PER_INPUT_TOKEN",
            output_token_envvar="TEST_COST_PER_OUTPUT_TOKEN",
            cost_per_input_token=0.01,
            cost_per_output_token=-0.02,
        )


def test_require_costs_accepts_zero_values():
    model_data = DeepEvalModelData(input_price=None, output_price=None)
    inp, out = require_costs(
        model_data,
        model_name="test-model",
        input_token_envvar="TEST_COST_PER_INPUT_TOKEN",
        output_token_envvar="TEST_COST_PER_OUTPUT_TOKEN",
        cost_per_input_token=0.0,
        cost_per_output_token=0.0,
    )
    assert inp == 0.0
    assert out == 0.0


def test_require_costs_partial_registry_falls_back_to_constructor():
    model_data = DeepEvalModelData(input_price=0.01, output_price=None)
    inp, out = require_costs(
        model_data,
        model_name="partial-model",
        input_token_envvar="TEST_COST_PER_INPUT_TOKEN",
        output_token_envvar="TEST_COST_PER_OUTPUT_TOKEN",
        cost_per_input_token=0.05,
        cost_per_output_token=0.10,
    )
    assert inp == 0.05
    assert out == 0.10
