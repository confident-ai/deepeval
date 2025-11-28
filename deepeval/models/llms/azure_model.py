from openai.types.chat.chat_completion import ChatCompletion
from openai import AzureOpenAI, AsyncAzureOpenAI
from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel, SecretStr

from deepeval.config.settings import get_settings
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.openai_model import (
    structured_outputs_models,
    json_mode_models,
    model_pricing,
)
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)

from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import parse_model_name, require_secret_api_key
from deepeval.constants import ProviderSlug as PS


retry_azure = create_retry_decorator(PS.AZURE)


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
        settings = get_settings()

        # fetch Azure deployment parameters
        model_name = model_name or settings.AZURE_MODEL_NAME
        self.deployment_name = deployment_name or settings.AZURE_DEPLOYMENT_NAME

        if azure_openai_api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.azure_openai_api_key: SecretStr | None = SecretStr(
                azure_openai_api_key
            )
        else:
            self.azure_openai_api_key = settings.AZURE_OPENAI_API_KEY

        self.openai_api_version = (
            openai_api_version or settings.OPENAI_API_VERSION
        )
        self.azure_endpoint = (
            azure_endpoint
            or settings.AZURE_OPENAI_ENDPOINT
            and str(settings.AZURE_OPENAI_ENDPOINT)
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
        if not async_mode:
            return self._build_client(AzureOpenAI)
        return self._build_client(AsyncAzureOpenAI)

    def _client_kwargs(self) -> Dict:
        """
        If Tenacity is managing retries, force OpenAI SDK retries off to avoid double retries.
        If the user opts into SDK retries for 'azure' via DEEPEVAL_SDK_RETRY_PROVIDERS,
        leave their retry settings as is.
        """
        kwargs = dict(self.kwargs or {})
        if not sdk_retries_for(PS.AZURE):
            kwargs["max_retries"] = 0
        return kwargs

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.azure_openai_api_key,
            provider_label="AzureOpenAI",
            env_var_name="AZURE_OPENAI_API_KEY",
            param_hint="`azure_openai_api_key` to AzureOpenAIModel(...)",
        )

        kw = dict(
            api_key=api_key,
            api_version=self.openai_api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.deployment_name,
            **self._client_kwargs(),
        )
        try:
            return cls(**kw)
        except TypeError as e:
            # older OpenAI SDKs may not accept max_retries, in that case remove and retry once
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise
