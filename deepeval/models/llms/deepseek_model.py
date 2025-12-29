from typing import Optional, Tuple, Union, Dict
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, SecretStr

from deepeval.errors import DeepEvalError
from deepeval.config.settings import get_settings
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import (
    require_costs,
    require_secret_api_key,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.constants import ProviderSlug as PS
from deepeval.models.llms.constants import DEEPSEEK_MODELS_DATA
from deepeval.utils import require_param


# consistent retry rules
retry_deepseek = create_retry_decorator(PS.DEEPSEEK)


class DeepSeekModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()

        model = model or settings.DEEPSEEK_MODEL_NAME

        if temperature is not None:
            temperature = float(temperature)
        elif settings.TEMPERATURE is not None:
            temperature = settings.TEMPERATURE
        else:
            temperature = 0.0

        cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else settings.DEEPSEEK_COST_PER_INPUT_TOKEN
        )
        cost_per_output_token = (
            cost_per_output_token
            if cost_per_output_token is not None
            else settings.DEEPSEEK_COST_PER_OUTPUT_TOKEN
        )

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.DEEPSEEK_API_KEY

        self.base_url = "https://api.deepseek.com"

        # validation
        model = require_param(
            model,
            provider_label="DeepSeekModel",
            env_var_name="DEEPSEEK_MODEL_NAME",
            param_hint="model",
        )

        if temperature < 0:
            raise DeepEvalError("Temperature must be >= 0.")

        self.model_data = DEEPSEEK_MODELS_DATA.get(model)
        self.temperature = temperature

        cost_per_input_token, cost_per_output_token = require_costs(
            self.model_data,
            model,
            "DEEPSEEK_COST_PER_INPUT_TOKEN",
            "DEEPSEEK_COST_PER_OUTPUT_TOKEN",
            cost_per_input_token,
            cost_per_output_token,
        )
        self.model_data.input_price = cost_per_input_token
        self.model_data.output_price = cost_per_output_token

        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = kwargs
        self.kwargs.pop("temperature", None)

        self.generation_kwargs = dict(generation_kwargs or {})
        self.generation_kwargs.pop("temperature", None)

        super().__init__(model)

    ###############################################
    # Other generate functions
    ###############################################

    @retry_deepseek
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:

        client = self.load_model(async_mode=False)
        if schema:
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
        else:
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
            return output, cost

    @retry_deepseek
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:

        client = self.load_model(async_mode=True)
        if schema:
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
        else:
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
            return output, cost

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        if self.model_data.input_price and self.model_data.output_price:
            input_cost = input_tokens * self.model_data.input_price
            output_cost = output_tokens * self.model_data.output_price
            return input_cost + output_cost

    ###############################################
    # Capabilities
    ###############################################

    def supports_log_probs(self) -> Union[bool, None]:
        return self.model_data.supports_log_probs

    def supports_temperature(self) -> Union[bool, None]:
        return self.model_data.supports_temperature

    def supports_multimodal(self) -> Union[bool, None]:
        return self.model_data.supports_multimodal

    def supports_structured_outputs(self) -> Union[bool, None]:
        return self.model_data.supports_structured_outputs

    def supports_json_mode(self) -> Union[bool, None]:
        return self.model_data.supports_json

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(OpenAI)
        return self._build_client(AsyncOpenAI)

    def _client_kwargs(self) -> Dict:
        kwargs = dict(self.kwargs or {})
        # if we are managing retries with Tenacity, force SDK retries off to avoid double retries.
        # if the user opts into SDK retries for "deepseek" via DEEPEVAL_SDK_RETRY_PROVIDERS, honor it.
        if not sdk_retries_for(PS.DEEPSEEK):
            kwargs["max_retries"] = 0
        return kwargs

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="DeepSeek",
            env_var_name="DEEPSEEK_API_KEY",
            param_hint="`api_key` to DeepSeekModel(...)",
        )

        kw = dict(
            api_key=api_key,
            base_url=self.base_url,
            **self._client_kwargs(),
        )
        try:
            return cls(**kw)
        except TypeError as e:
            # In case an older OpenAI client doesn’t accept max_retries, drop it and retry.
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise

    def get_model_name(self):
        return f"{self.name} (Deepseek)"
