from typing import Dict, Optional

from pydantic import AnyUrl, SecretStr

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.errors import DeepEvalError
from deepeval.models.llms.gateway_model import DeepEvalOpenAICompatibleModel
from deepeval.utils import require_param


class TheGridModel(DeepEvalOpenAICompatibleModel):
    """The Grid inference marketplace (https://thegrid.ai), reached through the OpenAI SDK.

    The Grid serves models from several labs behind one OpenAI-Chat-Completions
    compatible endpoint, so generation, structured outputs, retries and cost
    accounting all come from ``DeepEvalOpenAICompatibleModel``; this class only
    resolves configuration.

    Model names are *capability tiers* rather than a specific lab's model name —
    ``text-standard``, ``code-prime`` and ``agent-max`` each route to a current
    model for that tier. ``GET https://api.thegrid.ai/v1/models`` lists them.

    Like the other gateways, The Grid publishes no per-token price through its
    API, so cost is resolved from user-supplied ``cost_per_*_token`` values and
    is otherwise unknown.
    """

    PROVIDER_SLUG = PS.THEGRID
    PROVIDER_LABEL = "The Grid"
    API_KEY_ENV_VAR = "THEGRID_API_KEY"
    API_KEY_PARAM_HINT = "`api_key` to TheGridModel(...)"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[AnyUrl] = None,
        temperature: Optional[float] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()
        model = model or settings.THEGRID_MODEL_NAME

        if api_key is not None:
            # keep it secret, keep it safe from serializing, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.THEGRID_API_KEY

        if base_url is not None:
            base_url = str(base_url).rstrip("/")
        elif settings.THEGRID_BASE_URL is not None:
            base_url = str(settings.THEGRID_BASE_URL).rstrip("/")
        else:
            base_url = "https://api.thegrid.ai/v1"
        self.base_url = base_url

        # Instruments are capability tiers with no single sensible default, so
        # the tier must be chosen explicitly.
        model = require_param(
            model,
            provider_label="The Grid",
            env_var_name="THEGRID_MODEL_NAME",
            param_hint="model",
        )

        self.cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else settings.THEGRID_COST_PER_INPUT_TOKEN
        )
        self.cost_per_output_token = (
            cost_per_output_token
            if cost_per_output_token is not None
            else settings.THEGRID_COST_PER_OUTPUT_TOKEN
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
