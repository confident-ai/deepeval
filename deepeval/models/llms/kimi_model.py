from typing import Optional, Tuple, Union, Dict
from openai import OpenAI, AsyncOpenAI
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


retry_kimi = create_retry_decorator(PS.KIMI)

json_mode_models = [
    "kimi-thinking-preview",
    "kimi-k2-0711-preview",
    "kimi-latest-128k",
    "kimi-latest-32k",
    "kimi-latest-8k",
]

model_pricing = {
    "kimi-latest-8k": {
        "input": 0.20 / 1e6,
        "output": 2.00 / 1e6,
    },
    "kimi-latest-32k": {
        "input": 1.00 / 1e6,
        "output": 3.00 / 1e6,
    },
    "kimi-latest-128k": {
        "input": 2.00 / 1e6,
        "output": 5.00 / 1e6,
    },
    "kimi-k2-0711-preview": {
        "input": 0.60 / 1e6,
        "output": 2.50 / 1e6,
    },
    "kimi-thinking-preview": {
        "input": 30 / 1e6,
        "output": 30 / 1e6,
    },
    "moonshot-v1-8k": {
        "input": 1.00 / 1e6,
        "output": 2.00 / 1e6,
    },
    "moonshot-v1-32k": {
        "input": 2.00 / 1e6,
        "output": 3.00 / 1e6,
    },
    "moonshot-v1-128k": {
        "input": 0.20 / 1e6,
        "output": 5.00 / 1e6,
    },
    "moonshot-v1-8k-vision-preview": {
        "input": 1.00 / 1e6,
        "output": 2.00 / 1e6,
    },
    "moonshot-v1-32k-vision-preview": {
        "input": 2.00 / 1e6,
        "output": 3.00 / 1e6,
    },
    "moonshot-v1-128k-vision-preview": {
        "input": 0.20 / 1e6,
        "output": 5.00 / 1e6,
    },
}


class KimiModel(DeepEvalBaseLLM):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()

        model_name = model or settings.MOONSHOT_MODEL_NAME
        if model_name not in model_pricing:
            raise ValueError(
                f"Invalid model. Available Moonshot models: {', '.join(model_pricing.keys())}"
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
            self.api_key = settings.MOONSHOT_API_KEY

        self.base_url = "https://api.moonshot.cn/v1"
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_name)

    ###############################################
    # Other generate functions
    ###############################################

    @retry_kimi
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=False)
        if schema and self.model_name in json_mode_models:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=self.temperature,
                **self.generation_kwargs,
            )
            json_output = trim_and_load_json(
                completion.choices[0].message.content
            )
            cost = self.calculate_cost(
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            )
            return schema.model_validate(json_output), cost

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self.generation_kwargs,
        )
        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
        )
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return output, cost

    @retry_kimi
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=True)
        if schema and self.model_name in json_mode_models:
            completion = await client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=self.temperature,
                **self.generation_kwargs,
            )
            json_output = trim_and_load_json(
                completion.choices[0].message.content
            )
            cost = self.calculate_cost(
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            )
            return schema.model_validate(json_output), cost

        completion = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self.generation_kwargs,
        )
        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
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
        if not async_mode:
            return self._build_client(OpenAI)
        return self._build_client(AsyncOpenAI)

    def _client_kwargs(self) -> Dict:
        """
        If Tenacity is managing retries, force OpenAI SDK retries off to avoid double retries.
        If the user opts into SDK retries for 'kimi' via DEEPEVAL_SDK_RETRY_PROVIDERS,
        leave their retry settings as is.
        """
        kwargs = dict(self.kwargs or {})
        if not sdk_retries_for(PS.KIMI):
            kwargs["max_retries"] = 0
        return kwargs

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="Kimi",
            env_var_name="MOONSHOT_API_KEY",
            param_hint="`api_key` to KimiModel(...)",
        )

        kw = dict(
            api_key=api_key,
            base_url=self.base_url,
            **self._client_kwargs(),
        )
        try:
            return cls(**kw)
        except TypeError as e:
            # older OpenAI SDKs may not accept max_retries, in that case remove and retry once
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise

    def get_model_name(self):
        return f"{self.model_name}"
