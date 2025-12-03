from typing import Optional
from pydantic import SecretStr

from deepeval.errors import DeepEvalError


def parse_model_name(model_name: Optional[str] = None) -> str:
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

    if "/" in model_name:
        _, parsed_model_name = model_name.split("/", 1)
        return parsed_model_name
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
