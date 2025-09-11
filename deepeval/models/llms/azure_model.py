import logging
from tenacity import retry, before_sleep_log
from openai.types.chat.chat_completion import ChatCompletion
from openai import AzureOpenAI, AsyncAzureOpenAI
from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM
from deepeval.key_handler import ModelKeyValues, KEY_FILE_HANDLER
from deepeval.models.llms.openai_model import (
    structured_outputs_models,
    json_mode_models,
    model_pricing,
    log_retry_error,
)
from deepeval.models.retry_policy import (
    default_wait,
    default_stop,
    retry_predicate,
    AZURE_OPENAI_ERROR_POLICY,
)
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import parse_model_name


logger = logging.getLogger(__name__)

_base_retry_rules_kw = dict(
    wait=default_wait(),
    stop=default_stop(),
    retry=retry_predicate(AZURE_OPENAI_ERROR_POLICY),
    before_sleep=before_sleep_log(logger, logging.INFO),
    after=log_retry_error,
)
retry_azure = retry(**_base_retry_rules_kw)


class AzureOpenAIModel(DeepEvalBaseLLM):
    def __init__(
        self,
        deployment_name: Optional[str] = None,
        model_name: Optional[str] = None,
        azure_openai_api_key: Optional[str] = None,
        openai_api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        # fetch Azure deployment parameters
        model_name = model_name or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.AZURE_MODEL_NAME
        )
        self.deployment_name = deployment_name or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.AZURE_DEPLOYMENT_NAME
        )
        self.azure_openai_api_key = (
            azure_openai_api_key
            or KEY_FILE_HANDLER.fetch_data(ModelKeyValues.AZURE_OPENAI_API_KEY)
        )
        self.openai_api_version = (
            openai_api_version
            or KEY_FILE_HANDLER.fetch_data(ModelKeyValues.OPENAI_API_VERSION)
        )
        self.azure_endpoint = azure_endpoint or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.AZURE_OPENAI_ENDPOINT
        )
        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature

        # args and kwargs will be passed to the underlying model, in load_model function
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(parse_model_name(model_name))

    ###############################################
    # Other generate functions
    ###############################################

    @retry_azure
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        client = self.load_model(async_mode=False)
        if schema:
            if self.model_name in structured_outputs_models:
                completion = client.beta.chat.completions.parse(
                    model=self.deployment_name,
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
                    model=self.deployment_name,
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
            model=self.deployment_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            **self.generation_kwargs,
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

    @retry_azure
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        client = self.load_model(async_mode=True)
        if schema:
            if self.model_name in structured_outputs_models:
                completion = await client.beta.chat.completions.parse(
                    model=self.deployment_name,
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
                    model=self.deployment_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
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
            model=self.deployment_name,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
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
    # Other generate functions
    ###############################################

    @retry_azure
    def generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[ChatCompletion, float]:
        # Generate completion
        client = self.load_model(async_mode=False)
        completion = client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **self.generation_kwargs,
        )
        # Cost calculation
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)

        return completion, cost

    @retry_azure
    async def a_generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[ChatCompletion, float]:
        # Generate completion
        client = self.load_model(async_mode=True)
        completion = await client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **self.generation_kwargs,
        )
        # Cost calculation
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = self.calculate_cost(input_tokens, output_tokens)

        return completion, cost

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = model_pricing.get(self.model_name, model_pricing["gpt-4.1"])
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        return input_cost + output_cost

    ###############################################
    # Model
    ###############################################

    def get_model_name(self):
        return f"Azure OpenAI ({self.model_name})"

    def load_model(self, async_mode: bool = False):
        # ensure SDK retries are disabled and let Tenacity handle this via our retry policy
        kwargs = dict(self.kwargs or {})
        kwargs["max_retries"] = 0
        if not async_mode:
            return AzureOpenAI(
                api_key=self.azure_openai_api_key,
                api_version=self.openai_api_version,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.deployment_name,
                **kwargs,  # ← Keep this for client initialization
            )
        return AsyncAzureOpenAI(
            api_key=self.azure_openai_api_key,
            api_version=self.openai_api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.deployment_name,
            **kwargs,  # ← Keep this for client initialization
        )
