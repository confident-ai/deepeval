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
retry_cometapi = create_retry_decorator(PS.COMETAPI)

# CometAPI recommended models - 500+ AI Model API, All In One API
# Note: CometAPI pricing may vary; using placeholder values (0.0) for unknown models
# Users can override with cost_per_input_token and cost_per_output_token parameters
model_pricing = {
    # GPT series
    "gpt-5-chat-latest": {"input": 0.0, "output": 0.0},
    "gpt-5": {"input": 0.0, "output": 0.0},
    "gpt-5-pro": {"input": 0.0, "output": 0.0},
    "gpt-5-nano": {"input": 0.0, "output": 0.0},
    "gpt-4.1": {"input": 0.0, "output": 0.0},
    "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
    "o4-mini-2025-04-16": {"input": 0.0, "output": 0.0},
    "o3-pro-2025-06-10": {"input": 0.0, "output": 0.0},
    "chatgpt-4o-latest": {"input": 5.00 / 1e6, "output": 15.00 / 1e6},
    # Claude series
    "claude-sonnet-4-5-20250929": {"input": 0.0, "output": 0.0},
    "claude-opus-4-1-20250805": {"input": 0.0, "output": 0.0},
    "claude-opus-4-1-20250805-thinking": {"input": 0.0, "output": 0.0},
    "claude-sonnet-4-20250514": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-sonnet-4-20250514-thinking": {"input": 0.0, "output": 0.0},
    "claude-3-7-sonnet-latest": {"input": 3.00 / 1e6, "output": 15.00 / 1e6},
    "claude-3-5-haiku-latest": {"input": 1.00 / 1e6, "output": 5.00 / 1e6},
    # Gemini series
    "gemini-2.5-pro": {"input": 0.0, "output": 0.0},
    "gemini-2.5-flash": {"input": 0.0, "output": 0.0},
    "gemini-2.5-flash-lite": {"input": 0.0, "output": 0.0},
    "gemini-2.0-flash": {"input": 0.0, "output": 0.0},
    # Grok series
    "grok-4-0709": {"input": 0.0, "output": 0.0},
    "grok-4-fast-non-reasoning": {"input": 0.0, "output": 0.0},
    "grok-4-fast-reasoning": {"input": 0.0, "output": 0.0},
    # DeepSeek series
    "deepseek-v3.1": {"input": 0.0, "output": 0.0},
    "deepseek-v3": {"input": 0.27 / 1e6, "output": 1.10 / 1e6},
    "deepseek-r1-0528": {"input": 0.0, "output": 0.0},
    "deepseek-chat": {"input": 0.27 / 1e6, "output": 1.10 / 1e6},
    "deepseek-reasoner": {"input": 0.55 / 1e6, "output": 2.19 / 1e6},
    # Qwen series
    "qwen3-30b-a3b": {"input": 0.0, "output": 0.0},
    "qwen3-coder-plus-2025-07-22": {"input": 0.0, "output": 0.0},
}

# Default model for CometAPI
default_cometapi_model = "gpt-4o-mini"


class CometAPIModel(DeepEvalBaseLLM):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        model_name = model or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.COMETAPI_MODEL_NAME
        )
        
        # If no model is specified, use default
        if model_name is None:
            model_name = default_cometapi_model
        
        # Fetch cost overrides from key handler if not provided
        cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else KEY_FILE_HANDLER.fetch_data(
                ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN  # Reuse OpenAI cost keys
            )
        )
        cost_per_output_token = (
            cost_per_output_token
            if cost_per_output_token is not None
            else KEY_FILE_HANDLER.fetch_data(
                ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN
            )
        )
        
        # Add custom pricing if provided and model not in pricing dict
        if model_name not in model_pricing:
            if cost_per_input_token is not None and cost_per_output_token is not None:
                model_pricing[model_name] = {
                    "input": float(cost_per_input_token),
                    "output": float(cost_per_output_token),
                }
            else:
                # Allow unknown models with placeholder pricing
                model_pricing[model_name] = {
                    "input": 0.0,
                    "output": 0.0,
                }
        
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
            ModelKeyValues.COMETAPI_KEY
        )
        self.base_url = "https://api.cometapi.com/v1"
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_name)

    ###############################################
    # Generate functions
    ###############################################

    @retry_cometapi
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
                temperature=self.temperature,
                **self.generation_kwargs,
            )
            output = completion.choices[0].message.content
            cost = self.calculate_cost(
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            )
            return output, cost

    @retry_cometapi
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
                temperature=self.temperature,
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
        pricing = model_pricing.get(
            self.model_name,
            {"input": 0.0, "output": 0.0}
        )
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
        return f"CometAPI ({self.model_name})"

    def _client_kwargs(self) -> Dict:
        kwargs = dict(self.kwargs or {})
        # If we are managing retries with Tenacity, force SDK retries off to avoid double retries.
        # If the user opts into SDK retries for "cometapi" via DEEPEVAL_SDK_RETRY_PROVIDERS, honor it.
        if not sdk_retries_for(PS.COMETAPI):
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
            # In case an older OpenAI client doesn't accept max_retries, drop it and retry.
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise
