from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel, SecretStr

from deepeval.errors import DeepEvalError
from deepeval.config.settings import get_settings
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import (
    require_secret_api_key,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.constants import ProviderSlug as PS
from deepeval.models.llms.constants import GROK_MODELS_DATA
from deepeval.utils import require_param

# consistent retry rules
retry_grok = create_retry_decorator(PS.GROK)


class GrokModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):

        settings = get_settings()

        model = model or settings.GROK_MODEL_NAME

        if temperature is not None:
            temperature = float(temperature)
        elif settings.TEMPERATURE is not None:
            temperature = settings.TEMPERATURE
        else:
            temperature = 0.0

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: SecretStr | None = SecretStr(api_key)
        else:
            self.api_key = settings.GROK_API_KEY

        model = require_param(
            model,
            provider_label="GrokModel",
            env_var_name="GROK_MODEL_NAME",
            param_hint="model",
        )

        # validation
        if model not in GROK_MODELS_DATA.keys():
            raise DeepEvalError(
                f"Invalid model. Available Grok models: {', '.join(GROK_MODELS_DATA.keys())}"
            )

        if temperature < 0:
            raise DeepEvalError("Temperature must be >= 0.")

        self.model_data = GROK_MODELS_DATA.get(model)
        self.temperature = temperature

        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model)

    ###############################################
    # Other generate functions
    ###############################################

    @retry_grok
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:

        try:
            from xai_sdk.chat import user
        except ImportError:
            raise ImportError(
                "xai_sdk is required to use GrokModel. Please install it with: pip install xai-sdk"
            )
        client = self.load_model(async_mode=False)
        chat = client.chat.create(
            model=self.name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        chat.append(user(prompt))

        if schema and self.model_data.supports_structured_outputs:
            response, structured_output = chat.parse(schema)
            cost = self.calculate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            return structured_output, cost

        response = chat.sample()
        output = response.content
        cost = self.calculate_cost(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return output, cost

    @retry_grok
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:

        try:
            from xai_sdk.chat import user
        except ImportError:
            raise ImportError(
                "xai_sdk is required to use GrokModel. Please install it with: pip install xai-sdk"
            )
        client = self.load_model(async_mode=True)
        chat = client.chat.create(
            model=self.name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        chat.append(user(prompt))

        if schema and self.model_data.supports_structured_outputs:
            response, structured_output = await chat.parse(schema)
            cost = self.calculate_cost(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )
            return structured_output, cost

        response = await chat.sample()
        output = response.content
        cost = self.calculate_cost(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return output, cost

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        input_cost = input_tokens * self.model_data.input_price
        output_cost = output_tokens * self.model_data.output_price
        return input_cost + output_cost

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        try:
            from xai_sdk import Client, AsyncClient

            if not async_mode:
                return self._build_client(Client)
            else:
                return self._build_client(AsyncClient)
        except ImportError:
            raise ImportError(
                "xai_sdk is required to use GrokModel. Please install it with: pip install xai-sdk"
            )

    def _client_kwargs(self) -> Dict:
        """
        If Tenacity is managing retries, disable gRPC channel retries to avoid double retry.
        If the user opts into SDK retries for 'grok' via DEEPEVAL_SDK_RETRY_PROVIDERS,
        leave channel options as is
        """
        kwargs = dict(self.kwargs or {})
        opts = list(kwargs.get("channel_options", []))
        if not sdk_retries_for(PS.GROK):
            # remove any explicit enable flag, then disable retries
            opts = [
                option
                for option in opts
                if not (
                    isinstance(option, (tuple, list))
                    and option
                    and option[0] == "grpc.enable_retries"
                )
            ]
            opts.append(("grpc.enable_retries", 0))
        if opts:
            kwargs["channel_options"] = opts
        return kwargs

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="Grok",
            env_var_name="GROK_API_KEY",
            param_hint="`api_key` to GrokModel(...)",
        )

        kw = dict(api_key=api_key, **self._client_kwargs())
        try:
            return cls(**kw)
        except TypeError as e:
            # fallback: older SDK version might not accept channel_options
            if "channel_options" in str(e):
                kw.pop("channel_options", None)
                return cls(**kw)
            raise

    def get_model_name(self):
        return f"{self.name} (Grok)"
