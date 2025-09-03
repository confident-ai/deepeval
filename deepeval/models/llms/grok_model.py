from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel
import os

from deepeval.key_handler import ModelKeyValues, KEY_FILE_HANDLER
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models import DeepEvalBaseLLM


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
        "output": 2.50 / 1e-6,
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
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        model_name = model or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.GROK_MODEL_NAME
        )
        if model_name not in model_pricing:
            raise ValueError(
                f"Invalid model. Available Grok models: {', '.join(model_pricing.keys())}"
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
        self.api_key = (
            api_key
            or KEY_FILE_HANDLER.fetch_data(ModelKeyValues.GROK_API_KEY)
            or os.getenv("GROK_API_KEY")
        )
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_name)

    ###############################################
    # Other generate functions
    ###############################################

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
                return Client(api_key=self.api_key, **self.kwargs)
            else:
                return AsyncClient(api_key=self.api_key, **self.kwargs)
        except ImportError:
            raise ImportError(
                "xai_sdk is required to use GrokModel. Please install it with: pip install xai-sdk"
            )

    def get_model_name(self):
        return f"{self.model_name}"
