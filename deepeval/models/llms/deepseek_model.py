from typing import Optional, Tuple, Union, Dict
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

from deepeval.key_handler import ModelKeyValues, KEY_FILE_HANDLER
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.constants import ProviderSlug as PS


# consistent retry rules
retry_deepseek = create_retry_decorator(PS.DEEPSEEK)

model_pricing = {
    "deepseek-chat": {
        "input": 0.27 / 1e6,
        "output": 1.10 / 1e6,
    },
    "deepseek-reasoner": {
        "input": 0.55 / 1e6,
        "output": 2.19 / 1e6,
    },
}


class DeepSeekModel(DeepEvalBaseLLM):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        model_name = model or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.DEEPSEEK_MODEL_NAME
        )
        if model_name not in model_pricing:
            raise ValueError(
                f"Invalid model. Available DeepSeek models: {', '.join(model_pricing.keys())}"
            )
        temperature_from_key = KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.TEMPERATURE
        )
        if temperature_from_key is None:
            self.temperature = temperature
        else:
            self.temperature = float(temperature_from_key)
        if self.temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.api_key = api_key or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.DEEPSEEK_API_KEY
        )
        self.base_url = "https://api.deepseek.com"
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_name)

    ###############################################
    # Other generate functions
    ###############################################

    @retry_deepseek
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=False)
        if schema:
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
        else:
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
            return output, cost

    @retry_deepseek
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=True)
        if schema:
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
        else:
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

    def get_model_name(self):
        return f"{self.model_name}"

    def _client_kwargs(self) -> Dict:
        kwargs = dict(self.kwargs or {})
        # if we are managing retries with Tenacity, force SDK retries off to avoid double retries.
        # if the user opts into SDK retries for "deepseek" via DEEPEVAL_SDK_RETRY_PROVIDERS, honor it.
        if not sdk_retries_for(PS.DEEPSEEK):
            kwargs["max_retries"] = 0
        return kwargs

    def _build_client(self, cls):
        kw = dict(
            api_key=self.api_key,
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
