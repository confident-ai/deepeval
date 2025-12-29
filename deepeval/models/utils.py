import logging
from typing import Any, Dict, Optional, Tuple
from pydantic import SecretStr

from deepeval.errors import DeepEvalError


logger = logging.getLogger(__name__)


def parse_model_name(model_name: Optional[str] = None) -> Optional[str]:
    """Extract base model name from provider-prefixed format.

    This function is useful for extracting the actual model name from a
    provider-prefixed format which is used by some proxies like LiteLLM.
    LiteLLM is designed to work with many different LLM providers (OpenAI, Anthropic,
    Cohere, etc.). To tell it which provider's API to call, you prepend the provider
    name to the model ID, in the form "<provider>/<model>". So openai/gpt-4.1-mini
    literally means "OpenAI's GPT-4.1 Mini via the OpenAI chat completions endpoint."

    Args:
        model_name: Original model identifier, potentially in
            "<provider>/<model>" format

    Returns:
        The model name without provider prefix

    Examples:
        parse_model_name("openai/gpt-4o") -> "gpt-4o"
        parse_model_name("gpt-4o") -> "gpt-4o"
    """
    if model_name is None:
        return None

    # if "/" in model_name:
    #     _, parsed_model_name = model_name.split("/", 1)
    #     return parsed_model_name
    return model_name


def require_secret_api_key(
    secret: Optional[SecretStr],
    *,
    provider_label: str,
    env_var_name: str,
    param_hint: str,
) -> str:
    """
    Normalize and validate a provider API key stored as a SecretStr.

    Args:
        secret:
            The SecretStr coming from Settings or an explicit constructor arg.
        provider_label:
            Human readable provider name for error messages, such as Anthropic, or OpenAI etc
        env_var_name:
            The environment variable backing this key
        param_hint:
            A short hint telling users how to pass the key explicitly

    Returns:
        The underlying API key string.

    Raises:
        DeepEvalError: if the key is missing or empty.
    """
    if secret is None:
        raise DeepEvalError(
            f"{provider_label} API key is not configured. "
            f"Set {env_var_name} in your environment or pass "
            f"{param_hint}."
        )

    api_key = secret.get_secret_value()
    if not api_key:
        raise DeepEvalError(
            f"{provider_label} API key is empty. Please configure a valid key."
        )

    return api_key


def require_costs(
    model_data,
    model_name: str,
    input_token_envvar: str,
    output_token_envvar: str,
    cost_per_input_token: Optional[float] = None,
    cost_per_output_token: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Validates and returns the cost parameters (input and output tokens) for a model.

    Arguments:
    - model_data: The model's data object, which should contain `input_price` and `output_price`.
    - model_name: The model name used for error messaging.
    - cost_per_input_token: The input token cost provided during model initialization (optional).
    - cost_per_output_token: The output token cost provided during model initialization (optional).
    - input_token_envvar: The environment variable name for input cost.
    - output_token_envvar: The environment variable name for output cost.

    Returns:
    - A tuple of validated values (input_cost, output_cost). If the values are provided, they are returned.
      If not provided, they are fetched from settings or environment variables.
    """

    def validate_cost(
        value: Optional[float], envvar_name: str
    ) -> Optional[float]:
        """Helper function to validate the cost values."""
        if value is not None and value < 0:
            raise DeepEvalError(f"{envvar_name} must be >= 0.")
        return value

    # Validate provided token costs
    cost_per_input_token = validate_cost(
        cost_per_input_token, input_token_envvar
    )
    cost_per_output_token = validate_cost(
        cost_per_output_token, output_token_envvar
    )

    # If model data doesn't have pricing, use provided values or environment variables
    if model_data.input_price is None or model_data.output_price is None:
        if cost_per_input_token is None or cost_per_output_token is None:
            return None, None

        # Return the validated cost values as a tuple
        return cost_per_input_token, cost_per_output_token

    # If no custom cost values are provided, return model's default cost values
    return model_data.input_price, model_data.output_price


def normalize_kwargs_and_extract_aliases(
    provider_label: str,
    kwargs: Dict[str, Any],
    alias_map: Dict[str, list],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Normalize legacy keyword argument names according to alias_map.

    alias_map is of the form: {new_name: [old_name1, old_name2, ...]}

    - Returns (normalized_kwargs, extracted_values)
      where:
        - normalized_kwargs has all legacy keys removed (to prevent forwarding
          to downstream SDK clients).
        - extracted_values maps new_name -> value for any alias that was used.

    - Logs a warning for each legacy keyword used, so callers know they should
      migrate to the new name.
    """
    normalized = dict(kwargs)
    extracted: Dict[str, Any] = {}

    for new_name, old_names in alias_map.items():
        for old_name in old_names:
            if old_name in normalized:
                value = normalized.pop(old_name)

                logger.warning(
                    "%s keyword '%s' is deprecated; please use '%s' instead.",
                    provider_label,
                    old_name,
                    new_name,
                )

                # Only preserve the first alias value we see for a given new_name
                if new_name not in extracted:
                    extracted[new_name] = value

    return normalized, extracted
