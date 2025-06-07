from typing import Optional


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

    # if "/" in model_name:
    #     _, parsed_model_name = model_name.split("/", 1)
    #     return parsed_model_name
    return model_name
