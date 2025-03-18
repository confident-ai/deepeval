from tenacity import retry, retry_if_exception_type, wait_exponential_jitter
from openai import AzureOpenAI, AsyncAzureOpenAI
from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel
import openai
import json
import re

from deepeval.models import DeepEvalBaseLLM
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models.providers.gpt_model import (
    valid_gpt_models,
    structured_outputs_models,
    structured_outputs_models,
    json_mode_models,
    default_gpt_model,
    model_pricing,
    log_retry_error,
)


class AzureOpenAIModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        *args,
        **kwargs,
    ):
        model_name = model
        if isinstance(model, str) and model not in valid_gpt_models:
            raise ValueError(
                f"Invalid model. Available GPT models: {', '.join(model for model in valid_gpt_models)}"
            )
        elif model is None:
            model_name = default_gpt_model

        # fetch Azure deployment parameters
        self.azure_openai_api_key = KEY_FILE_HANDLER.fetch_data(
            KeyValues.AZURE_OPENAI_API_KEY
        )
        self.openai_api_version = KEY_FILE_HANDLER.fetch_data(
            KeyValues.OPENAI_API_VERSION
        )
        self.azure_deployment = KEY_FILE_HANDLER.fetch_data(
            KeyValues.AZURE_DEPLOYMENT_NAME
        )
        self.azure_endpoint = KEY_FILE_HANDLER.fetch_data(
            KeyValues.AZURE_OPENAI_ENDPOINT
        )

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
                )
                json_output = self.trim_and_load_json(
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
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            output = completion.choices[0].message.content
            cost = self.calculate_cost(
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            )
            return output, cost

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(openai.RateLimitError),
        after=log_retry_error,
    )
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        client = self.load_model(async_mode=True)
        if schema:
            client = AsyncAzureOpenAI(api_key=self.azure_openai_api_key)
            if self.model_name in structured_outputs_models:
                completion = await client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    response_format=schema,
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
                )
                json_output = self.trim_and_load_json(
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
                messages=[
                    {"role": "user", "content": prompt},
                ],
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

    def trim_and_load_json(
        self,
        input_string: str,
    ) -> Dict:
        start = input_string.find("{")
        end = input_string.rfind("}") + 1
        if end == 0 and start != -1:
            input_string = input_string + "}"
            end = len(input_string)
        jsonStr = input_string[start:end] if start != -1 and end != 0 else ""
        jsonStr = re.sub(r",\s*([\]}])", r"\1", jsonStr)
        try:
            return json.loads(jsonStr)
        except json.JSONDecodeError:
            error_str = "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
            raise ValueError(error_str)
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {str(e)}")

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = model_pricing.get(self.model_name, model_pricing["gpt-4o"])
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        return input_cost + output_cost

    ###############################################
    # Model
    ###############################################

    def get_model_name(self):
        return "Azure OpenAI"

    def load_model(self, async_mode: bool = False):
        if async_mode == False:
            return AzureOpenAI(
                api_key=self.azure_openai_api_key,
                api_version=self.openai_api_version,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_deployment,
            )
        else:
            return AsyncAzureOpenAI(
                api_key=self.azure_openai_api_key,
                api_version=self.openai_api_version,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_deployment,
            )
