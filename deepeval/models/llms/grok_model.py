from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel, SecretStr

from deepeval.config.settings import get_settings
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import require_secret_api_key
from deepeval.models import DeepEvalBaseLLM
from deepeval.constants import ProviderSlug as PS


# consistent retry rules
retry_grok = create_retry_decorator(PS.GROK)


structured_outputs_models = [
    "grok-4-0709",
    "grok-3",
    "grok-3-mini",
    "grok-3-fast",
    "grok-3-mini-fast",
]

model_pricing = {
    "grok-4-0709": {
        "input": 0.20 / 1e6,
        "output": 2.00 / 1e6,
    },
    "grok-3": {
        "input": 1.00 / 1e6,
        "output": 3.00 / 1e6,
    },
    "grok-3-mini": {
        "input": 2.00 / 1e6,
        "output": 5.00 / 1e6,
    },
    "grok-3-fast": {
        "input": 0.60 / 1e6,
        "output": 2.50 / 1e6,
    },
    "grok-3-mini-fast": {
        "input": 30 / 1e6,
        "output": 30 / 1e6,
    },
    "grok-2-vision-1212": {
        "input": 1.00 / 1e6,
        "output": 2.00 / 1e6,
    },
}


class GrokModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()

        model_name = model or settings.GROK_MODEL_NAME

        if model_name not in model_pricing:
            raise ValueError(
                f"Invalid model. Available Grok models: {', '.join(model_pricing.keys())}"
            )
        temperature_from_key = settings.TEMPERATURE
        if temperature_from_key is None:
            self.temperature = temperature
        else:
            self.temperature = float(temperature_from_key)
        if self.temperature < 0:
            raise ValueError("Temperature must be >= 0.")

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: SecretStr | None = SecretStr(api_key)
        else:
            self.api_key = settings.GROK_API_KEY

        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_name)

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
            model=self.model_name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        chat.append(user(prompt))

        if schema and self.model_name in structured_outputs_models:
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
            model=self.model_name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        chat.append(user(prompt))

        if schema and self.model_name in structured_outputs_models:
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
        pricing = model_pricing.get(self.model_name, model_pricing)
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
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

    def get_model_name(self):
        return f"{self.model_name}"

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
