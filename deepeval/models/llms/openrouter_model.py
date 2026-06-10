from typing import Dict, Optional

from pydantic import SecretStr

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.errors import DeepEvalError
from deepeval.models.llms.constants import DEFAULT_OPENROUTER_MODEL
from deepeval.models.llms.gateway_model import DeepEvalOpenAICompatibleModel


class OpenRouterModel(DeepEvalOpenAICompatibleModel):
    """OpenRouter gateway (https://openrouter.ai), reached through the OpenAI SDK.

    OpenRouter is OpenAI-Chat-Completions compatible, so all of the generation,
    structured-output, retry and cost logic comes from
    ``DeepEvalOpenAICompatibleModel``; this class only resolves configuration.
    """

    PROVIDER_SLUG = PS.OPENROUTER
    PROVIDER_LABEL = "OpenRouter"
    API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
    API_KEY_PARAM_HINT = "`api_key` to OpenRouterModel(...)"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()
        model = (
            model or settings.OPENROUTER_MODEL_NAME or DEFAULT_OPENROUTER_MODEL
        )

        if api_key is not None:
            # keep it secret, keep it safe from serializing, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.OPENROUTER_API_KEY

        if base_url is not None:
            base_url = str(base_url).rstrip("/")
        elif settings.OPENROUTER_BASE_URL is not None:
            base_url = str(settings.OPENROUTER_BASE_URL).rstrip("/")
        else:
            base_url = "https://openrouter.ai/api/v1"
        self.base_url = base_url

        self.cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else settings.OPENROUTER_COST_PER_INPUT_TOKEN
        )
        self.cost_per_output_token = (
            cost_per_output_token
            if cost_per_output_token is not None
            else settings.OPENROUTER_COST_PER_OUTPUT_TOKEN
        )

        if temperature is not None:
            temperature = float(temperature)
        elif settings.TEMPERATURE is not None:
            temperature = settings.TEMPERATURE
        else:
            temperature = 0.0
        if temperature < 0:
            raise DeepEvalError("Temperature must be >= 0.")
        self.temperature = temperature

        self.kwargs = dict(kwargs)
        self.kwargs.pop("temperature", None)

        self.generation_kwargs = dict(generation_kwargs or {})
        self.generation_kwargs.pop("temperature", None)

        super().__init__(model)
