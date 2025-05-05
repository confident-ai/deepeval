def get_actual_model_name(model_name: str) -> str:
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
        get_actual_model_name("openai/gpt-4o") -> "gpt-4o"
        get_actual_model_name("gpt-4o") -> "gpt-4o"
    """
    if "/" in model_name:
        _, model_name = model_name.split("/", 1)
        return model_name
    return model_name
