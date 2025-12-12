from typing import Optional, Tuple, Union, Dict
from openai import OpenAI, AsyncOpenAI
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
from deepeval.models.llms.constants import KIMI_MODELS_DATA
from deepeval.utils import require_param

retry_kimi = create_retry_decorator(PS.KIMI)


class KimiModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()

        model = model or settings.MOONSHOT_MODEL_NAME

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
            self.api_key = settings.MOONSHOT_API_KEY

        # validation
        model = require_param(
            model,
            provider_label="KimiModel",
            env_var_name="MOONSHOT_MODEL_NAME",
            param_hint="model",
        )

        if model not in KIMI_MODELS_DATA.keys():
            raise DeepEvalError(
                f"Invalid model. Available Moonshot models: {', '.join(KIMI_MODELS_DATA.keys())}"
            )

        if temperature < 0:
            raise DeepEvalError("Temperature must be >= 0.")

        self.model_data = KIMI_MODELS_DATA.get(model)
        self.temperature = temperature

        self.base_url = "https://api.moonshot.cn/v1"
        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model)

    ###############################################
    # Other generate functions
    ###############################################

    @retry_kimi
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:

        client = self.load_model(async_mode=False)
        if schema and self.model_data.supports_json:
            completion = client.chat.completions.create(
                model=self.name,
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
            model=self.name,
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
        if schema and self.model_data.supports_json:
            completion = await client.chat.completions.create(
                model=self.name,
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
            model=self.name,
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
        input_cost = input_tokens * self.model_data.input_price
        output_cost = output_tokens * self.model_data.output_price
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
        return f"{self.name} (KIMI)"
