from openai.types.chat.chat_completion import ChatCompletion
from deepeval.key_handler import ModelKeyValues, KEY_FILE_HANDLER
from typing import Optional, Tuple, Union, Dict
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from dataclasses import dataclass, field
from enum import Enum
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


@dataclass
class ModelCapabilities:
    """Defines the capabilities of a specific OpenAI model."""
    supports_structured_outputs: bool = False
    supports_json_mode: bool = False
    supports_log_probs: bool = True
    requires_temperature_1: bool = False
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None


@dataclass
class OpenAIModelConfig:
    """Configuration for an OpenAI model with its capabilities."""
    name: str
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)


class OpenAIModel(Enum):
    """Enumeration of supported OpenAI models with type safety and embedded configurations."""

    # GPT-3.5 models
    GPT_3_5_TURBO = OpenAIModelConfig(
        name="gpt-3.5-turbo",
        capabilities=ModelCapabilities(
            supports_json_mode=True,
            input_cost_per_token=0.50 / 1e6,
            output_cost_per_token=1.50 / 1e6
        )
    )
    GPT_3_5_TURBO_0125 = OpenAIModelConfig(
        name="gpt-3.5-turbo-0125",
        capabilities=ModelCapabilities(
            supports_json_mode=True,
            input_cost_per_token=0.50 / 1e6,
            output_cost_per_token=1.50 / 1e6
        )
    )
    GPT_3_5_TURBO_1106 = OpenAIModelConfig(
        name="gpt-3.5-turbo-1106",
        capabilities=ModelCapabilities(
            supports_json_mode=True,
            input_cost_per_token=1.00 / 1e6,
            output_cost_per_token=2.00 / 1e6
        )
    )

    # GPT-4 models
    GPT_4_0125_PREVIEW = OpenAIModelConfig(
        name="gpt-4-0125-preview",
        capabilities=ModelCapabilities(
            supports_json_mode=True,
            input_cost_per_token=10.00 / 1e6,
            output_cost_per_token=30.00 / 1e6
        )
    )
    GPT_4_1106_PREVIEW = OpenAIModelConfig(
        name="gpt-4-1106-preview",
        capabilities=ModelCapabilities(
            supports_json_mode=True,
            input_cost_per_token=10.00 / 1e6,
            output_cost_per_token=30.00 / 1e6
        )
    )
    GPT_4_TURBO = OpenAIModelConfig(
        name="gpt-4-turbo",
        capabilities=ModelCapabilities(
            supports_json_mode=True,
            input_cost_per_token=10.00 / 1e6,
            output_cost_per_token=30.00 / 1e6
        )
    )
    GPT_4_TURBO_2024_04_09 = OpenAIModelConfig(
        name="gpt-4-turbo-2024-04-09",
        capabilities=ModelCapabilities(
            supports_json_mode=True,
            input_cost_per_token=10.00 / 1e6,
            output_cost_per_token=30.00 / 1e6
        )
    )
    GPT_4_TURBO_PREVIEW = OpenAIModelConfig(
        name="gpt-4-turbo-preview",
        capabilities=ModelCapabilities(
            supports_json_mode=True,
            input_cost_per_token=10.00 / 1e6,
            output_cost_per_token=30.00 / 1e6
        )
    )
    GPT_4_32K = OpenAIModelConfig(
        name="gpt-4-32k",
        capabilities=ModelCapabilities(
            supports_json_mode=True,
            input_cost_per_token=60.00 / 1e6,
            output_cost_per_token=120.00 / 1e6
        )
    )
    GPT_4_32K_0613 = OpenAIModelConfig(
        name="gpt-4-32k-0613",
        capabilities=ModelCapabilities(
            supports_json_mode=True,
            input_cost_per_token=60.00 / 1e6,
            output_cost_per_token=120.00 / 1e6
        )
    )

    # GPT-4o models
    GPT_4O = OpenAIModelConfig(
        name="gpt-4o",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            input_cost_per_token=2.50 / 1e6,
            output_cost_per_token=10.00 / 1e6
        )
    )
    GPT_4O_2024_05_13 = OpenAIModelConfig(
        name="gpt-4o-2024-05-13",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            input_cost_per_token=2.50 / 1e6,
            output_cost_per_token=10.00 / 1e6
        )
    )
    GPT_4O_2024_08_06 = OpenAIModelConfig(
        name="gpt-4o-2024-08-06",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            input_cost_per_token=2.50 / 1e6,
            output_cost_per_token=10.00 / 1e6
        )
    )
    GPT_4O_2024_11_20 = OpenAIModelConfig(
        name="gpt-4o-2024-11-20",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            input_cost_per_token=2.50 / 1e6,
            output_cost_per_token=10.00 / 1e6
        )
    )
    GPT_4O_MINI = OpenAIModelConfig(
        name="gpt-4o-mini",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            input_cost_per_token=0.150 / 1e6,
            output_cost_per_token=0.600 / 1e6
        )
    )
    GPT_4O_MINI_2024_07_18 = OpenAIModelConfig(
        name="gpt-4o-mini-2024-07-18",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            input_cost_per_token=0.150 / 1e6,
            output_cost_per_token=0.600 / 1e6
        )
    )

    # GPT-4.1 models
    GPT_4_1 = OpenAIModelConfig(
        name="gpt-4.1",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            input_cost_per_token=2.00 / 1e6,
            output_cost_per_token=8.00 / 1e6
        )
    )
    GPT_4_1_MINI = OpenAIModelConfig(
        name="gpt-4.1-mini",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            input_cost_per_token=0.4 / 1e6,
            output_cost_per_token=1.60 / 1e6
        )
    )
    GPT_4_1_NANO = OpenAIModelConfig(
        name="gpt-4.1-nano",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            input_cost_per_token=0.1 / 1e6,
            output_cost_per_token=0.4 / 1e6
        )
    )

    # GPT-4.5 models
    GPT_4_5_PREVIEW = OpenAIModelConfig(
        name="gpt-4.5-preview",
        capabilities=ModelCapabilities(
            input_cost_per_token=75.00 / 1e6,
            output_cost_per_token=150.00 / 1e6
        )
    )
    GPT_4_5_PREVIEW_2025_02_27 = OpenAIModelConfig(
        name="gpt-4.5-preview-2025-02-27",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            supports_log_probs=False,
            input_cost_per_token=75.00 / 1e6,
            output_cost_per_token=150.00 / 1e6
        )
    )

    # O1 models
    O1 = OpenAIModelConfig(
        name="o1",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=15.00 / 1e6,
            output_cost_per_token=60.00 / 1e6
        )
    )
    O1_PREVIEW = OpenAIModelConfig(
        name="o1-preview",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            supports_log_probs=False,
            input_cost_per_token=15.00 / 1e6,
            output_cost_per_token=60.00 / 1e6
        )
    )
    O1_2024_12_17 = OpenAIModelConfig(
        name="o1-2024-12-17",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=15.00 / 1e6,
            output_cost_per_token=60.00 / 1e6
        )
    )
    O1_PREVIEW_2024_09_12 = OpenAIModelConfig(
        name="o1-preview-2024-09-12",
        capabilities=ModelCapabilities(
            supports_log_probs=False,
            input_cost_per_token=15.00 / 1e6,
            output_cost_per_token=60.00 / 1e6
        )
    )
    O1_MINI = OpenAIModelConfig(
        name="o1-mini",
        capabilities=ModelCapabilities(
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=3.00 / 1e6,
            output_cost_per_token=12.00 / 1e6
        )
    )
    O1_MINI_2024_09_12 = OpenAIModelConfig(
        name="o1-mini-2024-09-12",
        capabilities=ModelCapabilities(
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=3.00 / 1e6,
            output_cost_per_token=12.00 / 1e6
        )
    )

    # O3 models
    O3_MINI = OpenAIModelConfig(
        name="o3-mini",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=1.10 / 1e6,
            output_cost_per_token=4.40 / 1e6
        )
    )
    O3_MINI_2025_01_31 = OpenAIModelConfig(
        name="o3-mini-2025-01-31",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=1.10 / 1e6,
            output_cost_per_token=4.40 / 1e6
        )
    )

    # O4 models
    O4_MINI = OpenAIModelConfig(
        name="o4-mini",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=1.10 / 1e6,
            output_cost_per_token=4.40 / 1e6
        )
    )
    O4_MINI_2025_04_16 = OpenAIModelConfig(
        name="o4-mini-2025-04-16",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=1.10 / 1e6,
            output_cost_per_token=4.40 / 1e6
        )
    )

    # GPT-5 models
    GPT_5 = OpenAIModelConfig(
        name="gpt-5",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=1.25 / 1e6,
            output_cost_per_token=10.00 / 1e6
        )
    )
    GPT_5_2025_08_07 = OpenAIModelConfig(
        name="gpt-5-2025-08-07",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=1.25 / 1e6,
            output_cost_per_token=10.00 / 1e6
        )
    )
    GPT_5_MINI = OpenAIModelConfig(
        name="gpt-5-mini",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=0.25 / 1e6,
            output_cost_per_token=2.00 / 1e6
        )
    )
    GPT_5_MINI_2025_08_07 = OpenAIModelConfig(
        name="gpt-5-mini-2025-08-07",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=0.25 / 1e6,
            output_cost_per_token=2.00 / 1e6
        )
    )
    GPT_5_NANO = OpenAIModelConfig(
        name="gpt-5-nano",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=0.05 / 1e6,
            output_cost_per_token=0.40 / 1e6
        )
    )
    GPT_5_NANO_2025_08_07 = OpenAIModelConfig(
        name="gpt-5-nano-2025-08-07",
        capabilities=ModelCapabilities(
            supports_structured_outputs=True,
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=0.05 / 1e6,
            output_cost_per_token=0.40 / 1e6
        )
    )
    GPT_5_CHAT_LATEST = OpenAIModelConfig(
        name="gpt-5-chat-latest",
        capabilities=ModelCapabilities(
            supports_log_probs=False,
            requires_temperature_1=True,
            input_cost_per_token=1.25 / 1e6,
            output_cost_per_token=10.00 / 1e6
        )
    )

    @classmethod
    def from_name(cls, name: str) -> Optional['OpenAIModel']:
        """Get OpenAIModel enum from model name string."""
        for model in cls:
            if model.value.name == name:
                return model
        return None


# Default model
default_gpt_model = "gpt-4.1"

# Helper functions for backward compatibility
def get_model_config(model_name: str) -> Optional[OpenAIModelConfig]:
    """Get model configuration from enum or return None if not found."""
    model_enum = OpenAIModel.from_name(model_name)
    return model_enum.value if model_enum else None

# Backward compatibility - create MODEL_REGISTRY from enum values
MODEL_REGISTRY: Dict[str, OpenAIModelConfig] = {
    model.value.name: model.value for model in OpenAIModel
}


retryable_exceptions = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.LengthFinishReasonError,
)


# Helper functions for backward compatibility
def get_model_capabilities(model_name: str) -> ModelCapabilities:
    """Get capabilities for a model, returning defaults if not in registry."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name].capabilities
    return ModelCapabilities()

# Export lists for backward compatibility
structured_outputs_models = [
    name for name, config in MODEL_REGISTRY.items()
    if config.capabilities.supports_structured_outputs
]

json_mode_models = [
    name for name, config in MODEL_REGISTRY.items()
    if config.capabilities.supports_json_mode
]

unsupported_log_probs_gpt_models = [
    name for name, config in MODEL_REGISTRY.items()
    if not config.capabilities.supports_log_probs
]

model_pricing = {
    name: {
        "input": config.capabilities.input_cost_per_token or 0,
        "output": config.capabilities.output_cost_per_token or 0
    }
    for name, config in MODEL_REGISTRY.items()
    if config.capabilities.input_cost_per_token is not None
}


class GPTModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[Union[str, OpenAIModel]] = None,
        _openai_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        temperature: float = 0,
        supports_structured_outputs: Optional[bool] = None,
        supports_json_mode: Optional[bool] = None,
        supports_log_probs: Optional[bool] = None,
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

        # Handle both enum and string model inputs
        if model:
            model_name = parse_model_name(model)
            # Try to find in enum first
            model_enum = OpenAIModel.from_name(model_name)
            if model_enum:
                self.model_config = model_enum.value
            else:
                # Create a default configuration for unknown models
                self.model_config = OpenAIModelConfig(
                    name=model_name,
                    capabilities=ModelCapabilities(
                        supports_structured_outputs=supports_structured_outputs or False,
                        supports_json_mode=supports_json_mode or False,
                        supports_log_probs=supports_log_probs if supports_log_probs is not None else True,
                        input_cost_per_token=cost_per_input_token,
                        output_cost_per_token=cost_per_output_token
                    )
                )
                logging.info(f"Using model '{model_name}' not in registry. Using provided or default capabilities.")
        elif model is None:
            model_name = default_gpt_model
            # Default to gpt-4.1
            self.model_config = OpenAIModel.GPT_4_1.value

        # Override capabilities if explicitly provided
        if supports_structured_outputs is not None:
            self.model_config.capabilities.supports_structured_outputs = supports_structured_outputs
        if supports_json_mode is not None:
            self.model_config.capabilities.supports_json_mode = supports_json_mode
        if supports_log_probs is not None:
            self.model_config.capabilities.supports_log_probs = supports_log_probs

        # Handle pricing
        if self.model_config.capabilities.input_cost_per_token is None or self.model_config.capabilities.output_cost_per_token is None:
            if cost_per_input_token is None or cost_per_output_token is None:
                raise ValueError(
                    f"No pricing available for `{model_name}`. "
                    "Please provide both `cost_per_input_token` and `cost_per_output_token` when initializing `GPTModel`, "
                    "or set them via the CLI:\n"
                    "    deepeval set-openai --model=[...] --cost_per_input_token=[...] --cost_per_output_token=[...]"
                )
            else:
                self.model_config.capabilities.input_cost_per_token = float(cost_per_input_token)
                self.model_config.capabilities.output_cost_per_token = float(cost_per_output_token)

        self._openai_api_key = _openai_api_key
        self.base_url = base_url

        # Auto-adjust temperature for models that require it
        if self.model_config.capabilities.requires_temperature_1:
            temperature = 1

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
            if self.model_config.capabilities.supports_structured_outputs:
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
            if self.model_config.capabilities.supports_json_mode:
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
            if self.model_config.capabilities.supports_structured_outputs:
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
            if self.model_config.capabilities.supports_json_mode:
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
        completion_kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        # Only add logprobs if the model supports it
        if self.model_config.capabilities.supports_log_probs:
            completion_kwargs["logprobs"] = True
            completion_kwargs["top_logprobs"] = top_logprobs

        completion = client.chat.completions.create(**completion_kwargs)
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
        completion_kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        # Only add logprobs if the model supports it
        if self.model_config.capabilities.supports_log_probs:
            completion_kwargs["logprobs"] = True
            completion_kwargs["top_logprobs"] = top_logprobs

        completion = await client.chat.completions.create(**completion_kwargs)
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
        input_cost = input_tokens * (self.model_config.capabilities.input_cost_per_token or 0)
        output_cost = output_tokens * (self.model_config.capabilities.output_cost_per_token or 0)
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
