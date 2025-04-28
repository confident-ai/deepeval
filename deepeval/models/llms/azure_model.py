from typing import Dict, Optional, Tuple, Union

import openai
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI
from openai import AsyncAzureOpenAI, AzureOpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter

from deepeval.key_handler import KEY_FILE_HANDLER, KeyValues
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.openai_model import (
    json_mode_models,
    log_retry_error,
    model_pricing,
    structured_outputs_models,
)
from deepeval.models.llms.utils import trim_and_load_json

retryable_exceptions = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.LengthFinishReasonError,
)


class AzureOpenAIModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: Optional[str] = None,
        deploynment_name: Optional[str] = None,
        azure_openai_api_key: Optional[str] = None,
        openai_api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        temperature: float = 0,
        *args,
        **kwargs,
    ):
        # fetch Azure deployment parameters
        model_name = model_name or KEY_FILE_HANDLER.fetch_data(
            KeyValues.AZURE_MODEL_NAME
        )
        self.deploynment_name = deploynment_name or KEY_FILE_HANDLER.fetch_data(
            KeyValues.AZURE_DEPLOYMENT_NAME
        )
        self.azure_openai_api_key = (
            azure_openai_api_key
            or KEY_FILE_HANDLER.fetch_data(KeyValues.AZURE_OPENAI_API_KEY)
        )
        self.openai_api_version = (
            openai_api_version
            or KEY_FILE_HANDLER.fetch_data(KeyValues.OPENAI_API_VERSION)
        )
        self.azure_endpoint = azure_endpoint or KEY_FILE_HANDLER.fetch_data(
            KeyValues.AZURE_OPENAI_ENDPOINT
        )
        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature

        # args and kwargs will be passed to the underlying model, in load_model function
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    ###############################################
    # Other generate functions
    ###############################################

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
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
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
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

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
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
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
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

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate_raw_response(
        self, prompt: str, **kwargs
    ) -> Tuple[AIMessage, float]:
        chat_model = self.load_langchain_model().bind(**kwargs)
        with get_openai_callback() as cb:
            res = chat_model.invoke(prompt)
            return res, cb.total_cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    async def a_generate_raw_response(
        self, prompt: str, **kwargs
    ) -> Tuple[AIMessage, float]:
        chat_model = self.load_langchain_model().bind(**kwargs)
        with get_openai_callback() as cb:
            res = await chat_model.ainvoke(prompt)
        return res, cb.total_cost

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = model_pricing.get(self.model_name, model_pricing["gpt-4o"])
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        return input_cost + output_cost

    ###############################################
    # Model
    ###############################################

    def get_model_name(self):
        return f"Azure OpenAI ({self.model_name})"

    def load_model(self, async_mode: bool = False):
        if async_mode == False:
            return AzureOpenAI(
                api_key=self.azure_openai_api_key,
                api_version=self.openai_api_version,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.deploynment_name,
            )
        else:
            return AsyncAzureOpenAI(
                api_key=self.azure_openai_api_key,
                api_version=self.openai_api_version,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.deploynment_name,
            )

    def load_langchain_model(self):
        return AzureChatOpenAI(
            azure_deployment=self.deploynment_name,
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_openai_api_key,
            api_version=self.openai_api_version,
            temperature=self.temperature,
            *self.args,
            **self.kwargs,
        )
