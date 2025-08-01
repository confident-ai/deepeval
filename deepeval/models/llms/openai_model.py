from openai.types.chat.chat_completion import ChatCompletion
from deepeval.key_handler import ModelKeyValues, KEY_FILE_HANDLER
from typing import Optional, Tuple, Union, Dict
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
import logging
import openai

from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential_jitter,
    RetryCallState,
)

from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import parse_model_name


def log_retry_error(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    logging.error(
        f"OpenAI Error: {exception} Retrying: {retry_state.attempt_number} time(s)..."
    )


valid_gpt_models = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4.5-preview",
    "o1",
    "o1-preview",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    "o4-mini",
    "gpt-4.5-preview-2025-02-27",
]

unsupported_log_probs_gpt_models = [
    "o1",
    "o1-preview",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    "gpt-4.5-preview-2025-02-27",
]

structured_outputs_models = [
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o1",
    "o1-preview",
    "o1-2024-12-17",
    "o3-mini",
    "o3-mini-2025-01-31",
    "gpt-4.5-preview-2025-02-27",
]

json_mode_models = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-1106",
    "gpt-4-0125-preview",
    "gpt-4-1106-preview",
    "gpt-4-turbo",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo-preview",
    "gpt-4-32k",
    "gpt-4-32k-0613",
]

model_pricing = {
    "gpt-4o-mini": {"input": 0.150 / 1e6, "output": 0.600 / 1e6},
    "gpt-4o": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
    "gpt-4-turbo": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "gpt-4-turbo-preview": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "gpt-4-0125-preview": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "gpt-4-1106-preview": {"input": 10.00 / 1e6, "output": 30.00 / 1e6},
    "gpt-4": {"input": 30.00 / 1e6, "output": 60.00 / 1e6},
    "gpt-4-32k": {"input": 60.00 / 1e6, "output": 120.00 / 1e6},
    "gpt-3.5-turbo-1106": {"input": 1.00 / 1e6, "output": 2.00 / 1e6},
    "gpt-3.5-turbo": {"input": 0.50 / 1e6, "output": 1.50 / 1e6},
    "gpt-3.5-turbo-16k": {"input": 3.00 / 1e6, "output": 4.00 / 1e6},
    "gpt-3.5-turbo-0125": {"input": 0.50 / 1e6, "output": 1.50 / 1e6},
    "gpt-3.5-turbo-instruct": {"input": 1.50 / 1e6, "output": 2.00 / 1e6},
    "o1": {"input": 15.00 / 1e6, "output": 60.00 / 1e6},
    "o1-preview": {"input": 15.00 / 1e6, "output": 60.00 / 1e6},
    "o1-2024-12-17": {"input": 15.00 / 1e6, "output": 60.00 / 1e6},
    "o3-mini": {"input": 1.10 / 1e6, "output": 4.40 / 1e6},
    "o3-mini-2025-01-31": {"input": 1.10 / 1e6, "output": 4.40 / 1e6},
    "o4-mini": {"input": 1.10 / 1e6, "output": 4.40 / 1e6},
    "gpt-4.1": {
        "input": 2.00 / 1e6,
        "output": 8.00 / 1e6,
    },
    "gpt-4.1-mini": {
        "input": 0.4 / 1e6,
        "output": 1.60 / 1e6,
    },
    "gpt-4.1-nano": {
        "input": 0.1 / 1e6,
        "output": 0.4 / 1e6,
    },
    "gpt-4.5-preview": {
        "input": 75.00 / 1e6,
        "output": 150.00 / 1e6,
    },
}

default_gpt_model = "gpt-4.1"

retryable_exceptions = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.LengthFinishReasonError,
)


class GPTModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        _openai_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        temperature: float = 0,
        **kwargs,
    ):
        model_name = None
        model = model or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.OPENAI_MODEL_NAME
        )
        cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else KEY_FILE_HANDLER.fetch_data(
                ModelKeyValues.OPENAI_COST_PER_INPUT_TOKEN
            )
        )
        cost_per_output_token = (
            cost_per_output_token
            if cost_per_output_token is not None
            else KEY_FILE_HANDLER.fetch_data(
                ModelKeyValues.OPENAI_COST_PER_OUTPUT_TOKEN
            )
        )

        if isinstance(model, str):
            model_name = parse_model_name(model)
            if model_name not in valid_gpt_models:
                raise ValueError(
                    f"Invalid model. Available GPT models: {', '.join(model for model in valid_gpt_models)}"
                )
        elif model is None:
            model_name = default_gpt_model

        if model_name not in model_pricing:
            if cost_per_input_token is None or cost_per_output_token is None:
                raise ValueError(
                    f"No pricing available for `{model_name}`. "
                    "Please provide both `cost_per_input_token` and `cost_per_output_token` when initializing `GPTModel`, "
                    "or set them via the CLI:\n"
                    "    deepeval set-openai --model=[...] --cost_per_input_token=[...] --cost_per_output_token=[...]"
                )
            else:
                model_pricing[model_name] = {
                    "input": float(cost_per_input_token),
                    "output": float(cost_per_output_token),
                }

        elif model is None:
            model_name = default_gpt_model

        self._openai_api_key = _openai_api_key
        self.base_url = base_url
        # args and kwargs will be passed to the underlying model, in load_model function
        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature
        self.kwargs = kwargs
        super().__init__(model_name)

    ###############################################
    # Generate functions
    ###############################################

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=False)
        if schema:
            if self.model_name in structured_outputs_models:
                completion = client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    response_format=schema,
                    temperature=self.temperature,
                )
                structured_output: BaseModel = completion.choices[
                    0
                ].message.parsed
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                return structured_output, cost
            if self.model_name in json_mode_models:
                completion = client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
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
            temperature=self.temperature,
        )
        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens, completion.usage.completion_tokens
        )
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return output, cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        client = self.load_model(async_mode=True)
        if schema:
            if self.model_name in structured_outputs_models:
                completion = await client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    response_format=schema,
                    temperature=self.temperature,
                )
                structured_output: BaseModel = completion.choices[
                    0
                ].message.parsed
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                return structured_output, cost
            if self.model_name in json_mode_models:
                completion = await client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
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
            temperature=self.temperature,
        )
        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens, completion.usage.completion_tokens
        )
        if schema:
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return output, cost

    ###############################################
    # Other generate functions
    ###############################################

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[ChatCompletion, float]:
        # Generate completion
        client = self.load_model(async_mode=False)
        completion = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        # Cost calculation
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)

        return completion, cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    async def a_generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[ChatCompletion, float]:
        # Generate completion
        client = self.load_model(async_mode=True)
        completion = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        # Cost calculation
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)

        return completion, cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[list[str], float]:
        client = self.load_model(async_mode=False)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            temperature=temperature,
        )
        completions = [choice.message.content for choice in response.choices]
        return completions

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = model_pricing.get(self.model_name, model_pricing)
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        return input_cost + output_cost

    ###############################################
    # Model
    ###############################################

    def get_model_name(self):
        return self.model_name

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return OpenAI(
                api_key=self._openai_api_key,
                base_url=self.base_url,
                **self.kwargs,
            )
        return AsyncOpenAI(
            api_key=self._openai_api_key, base_url=self.base_url, **self.kwargs
        )
