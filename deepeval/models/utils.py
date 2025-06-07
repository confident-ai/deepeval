from typing import Optional


def parse_model_name(model_name: Optional[str] = None) -> Optional[str]:
    """Extract base model name from provider-prefixed format.

    Some model serving systems (e.g., LiteLLM) prepend the provider to the model ID
    in the format "<provider>/<model>". This function strips the known provider
    prefix and returns the actual model name.

    For example, "openai/gpt-4o" means GPT-4o served via OpenAI. In such cases,
    we strip the "openai/" prefix.

    However, not all prefixes are providers — for example, "local/llama-3" might refer
    to a valid model path in a self-hosted setup. We preserve model names unless the
    prefix matches a known provider.

    Args:
        model_name: The original model identifier, potentially in "<provider>/<model>" format.

    Returns:
        The model name without provider prefix for known providers.
        Returns the full name for unknown providers (e.g., "local", "custom").

    Examples:
        parse_model_name("openai/gpt-4o") -> "gpt-4o"
        parse_model_name("gpt-4o") -> "gpt-4o"
        parse_model_name("local/llama-3") -> "local/llama-3"
        parse_model_name(None) -> None
    """
    if model_name is None:
        return None

    known_providers = {"openai", "anthropic", "cohere", "mistral", "groq", "huggingface"}

    if "/" in model_name:
        provider, parsed_model_name = model_name.split("/", 1)
        if provider in known_providers:
            return parsed_model_name
        else:
            return model_name

    return model_name
