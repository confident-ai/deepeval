from deepeval.models.llms.litellm_model import LiteLLMModel


class GatewayModel(LiteLLMModel):
    """Any provider's model, addressed as ``"<provider>/<model>"``.

    One class for every provider, so nothing provider-specific has to be
    learned or maintained: the provider's own API key is read from its usual
    environment variable, and keyword arguments such as ``reasoning_effort``
    reach the provider as given.

        GatewayModel("anthropic/claude-opus-5", temperature=0)
        GatewayModel("openai/gpt-5.4", reasoning_effort="high")

    Requires LiteLLM, which is not installed with deepeval:
    ``pip install litellm``.
    """
