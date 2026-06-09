from typing import Dict, Optional

from pydantic import AnyUrl, SecretStr

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.errors import DeepEvalError
from deepeval.models.llms.gateway_model import DeepEvalOpenAICompatibleModel
from deepeval.models.utils import require_secret_api_key
from deepeval.utils import require_param


class PortkeyModel(DeepEvalOpenAICompatibleModel):
    """Portkey AI gateway (https://portkey.ai), reached through the OpenAI SDK.

    Portkey exposes an OpenAI-Chat-Completions compatible endpoint, so generation,
    structured outputs, retries and cost accounting all come from
    ``DeepEvalOpenAICompatibleModel``. Portkey authenticates with its own headers
    (``x-portkey-api-key`` / ``x-portkey-provider``) rather than a bearer token,
    which is the only transport-level difference handled here.
    """

    PROVIDER_SLUG = PS.PORTKEY
    PROVIDER_LABEL = "Portkey"
    API_KEY_ENV_VAR = "PORTKEY_API_KEY"
    API_KEY_PARAM_HINT = "`api_key` to PortkeyModel(...)"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[AnyUrl] = None,
        provider: Optional[str] = None,
        temperature: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()
        model = model or settings.PORTKEY_MODEL_NAME

        if api_key is not None:
            # keep it secret, keep it safe from serializing, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.PORTKEY_API_KEY

        if base_url is not None:
            base_url = str(base_url).rstrip("/")
        elif settings.PORTKEY_BASE_URL is not None:
            base_url = str(settings.PORTKEY_BASE_URL).rstrip("/")

        provider = provider or settings.PORTKEY_PROVIDER_NAME

        # Portkey requires all three explicitly (no sensible default exists).
        model = require_param(
            model,
            provider_label="Portkey",
            env_var_name="PORTKEY_MODEL_NAME",
            param_hint="model",
        )
        self.base_url = require_param(
            base_url,
            provider_label="Portkey",
            env_var_name="PORTKEY_BASE_URL",
            param_hint="base_url",
        )
        self.provider = require_param(
            provider,
            provider_label="Portkey",
            env_var_name="PORTKEY_PROVIDER_NAME",
            param_hint="provider",
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

    def _client_extra_kwargs(self) -> Dict:
        """Inject Portkey's auth headers onto the OpenAI client.

        Portkey reads ``x-portkey-api-key`` (and an optional ``x-portkey-provider``)
        rather than the bearer token the OpenAI SDK sends by default. Any
        user-supplied ``default_headers`` are preserved.
        """
        api_key = require_secret_api_key(
            self.api_key,
            provider_label=self.PROVIDER_LABEL,
            env_var_name=self.API_KEY_ENV_VAR,
            param_hint=self.API_KEY_PARAM_HINT,
        )
        headers = dict((self.kwargs or {}).get("default_headers", {}))
        headers["x-portkey-api-key"] = api_key
        if self.provider:
            headers["x-portkey-provider"] = self.provider
        return {"default_headers": headers}
