from typing import Optional, Tuple, Union, Dict, List
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel

from deepeval.key_handler import ModelKeyValues, KEY_FILE_HANDLER
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models import DeepEvalBaseLLM


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
        "output": 2.50 / 1e-6,
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
        model_name = model or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.MOONSHOT_MODEL_NAME
        )
        if model_name not in model_pricing:
            raise ValueError(
                f"Invalid model. Available Moonshot models: {', '.join(model_pricing.keys())}"
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
            ModelKeyValues.MOONSHOT_API_KEY
        )
        self.base_url = "https://api.moonshot.cn/v1"
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_name)

    ###############################################
    # Other generate functions
    ###############################################

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

    def chat_generate(
        self, messages: List[Dict[str, str]], schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=False)
        if schema and self.model_name in json_mode_models:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
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
            messages=messages,
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

    async def a_chat_generate(
        self, messages: List[Dict[str, str]], schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=True)
        if schema and self.model_name in json_mode_models:
            completion = await client.chat.completions.create(
                model=self.model_name,
                messages=messages,
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
            messages=messages,
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
            return OpenAI(
                api_key=self.api_key, base_url=self.base_url, **self.kwargs
            )
        else:
            return AsyncOpenAI(
                api_key=self.api_key, base_url=self.base_url, **self.kwargs
            )

    def get_model_name(self):
        return f"{self.model_name}"
