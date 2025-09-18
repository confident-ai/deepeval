import warnings

from typing import Optional, Tuple, Union, Dict
from anthropic import Anthropic, AsyncAnthropic
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models.utils import parse_model_name
from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS


# consistent retry rules
retry_anthropic = create_retry_decorator(PS.ANTHROPIC)

model_pricing = {
    "claude-opus-4-20250514": {"input": 15.00 / 1e6, "output": 75.00 / 1e6},
    "claude-sonnet-4-20250514": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-3-7-sonnet-latest": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-3-5-haiku-latest": {"input": 0.80 / 1e6, "output": 4.00 / 1e6},
    "claude-3-5-sonnet-latest": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-3-opus-latest": {"input": 15.00 / 1e6, "output": 75.00 / 1e6},
    "claude-3-sonnet-20240229": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-3-haiku-20240307": {"input": 0.25 / 1e6, "output": 1.25 / 1e6},
    "claude-instant-1.2": {"input": 0.80 / 1e6, "output": 2.40 / 1e6},
}


class AnthropicModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = "claude-3-7-sonnet-latest",
        temperature: float = 0,
        _anthropic_api_key: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        model_name = parse_model_name(model)
        self._anthropic_api_key = _anthropic_api_key

        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature

        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_name)

    ###############################################
    # Generate functions
    ###############################################

    @retry_anthropic
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        chat_model = self.load_model()
        message = chat_model.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        cost = self.calculate_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )
        if schema is None:
            return message.content[0].text, cost
        else:
            json_output = trim_and_load_json(message.content[0].text)
            return schema.model_validate(json_output), cost

    @retry_anthropic
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        chat_model = self.load_model(async_mode=True)
        message = await chat_model.messages.create(
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        cost = self.calculate_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )
        if schema is None:
            return message.content[0].text, cost
        else:
            json_output = trim_and_load_json(message.content[0].text)

            return schema.model_validate(json_output), cost

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = model_pricing.get(self.model_name)

        if pricing is None:
            # Calculate average cost from all known models
            avg_input_cost = sum(
                p["input"] for p in model_pricing.values()
            ) / len(model_pricing)
            avg_output_cost = sum(
                p["output"] for p in model_pricing.values()
            ) / len(model_pricing)
            pricing = {"input": avg_input_cost, "output": avg_output_cost}

            warnings.warn(
                f"[Warning] Pricing not defined for model '{self.model_name}'. "
                "Using average input/output token costs from existing model_pricing."
            )

        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        return input_cost + output_cost

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(Anthropic)
        return self._build_client(AsyncAnthropic)

    def get_model_name(self):
        return f"{self.model_name}"

    def _client_kwargs(self) -> Dict:
        kwargs = dict(self.kwargs or {})
        # If we are managing retries with Tenacity, force SDK retries off to avoid double retries.
        # if the user opts into SDK retries via DEEPEVAL_SDK_RETRY_PROVIDERS, then honor their max_retries.
        if not sdk_retries_for(PS.ANTHROPIC):
            kwargs["max_retries"] = 0
        return kwargs

    def _build_client(self, cls):
        settings = get_settings()
        kw = dict(
            api_key=settings.ANTHROPIC_API_KEY or self._anthropic_api_key,
            **self._client_kwargs(),
        )
        try:
            return cls(**kw)
        except TypeError as e:
            # in case older SDKs don’t accept max_retries, drop it and retry
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise
