from openai.types.chat.chat_completion import ChatCompletion
from openai import AzureOpenAI, AsyncAzureOpenAI
from typing import Optional, Tuple, Union, Dict, List, Callable, Awaitable
from pydantic import BaseModel, SecretStr

from deepeval.errors import DeepEvalError
from deepeval.config.settings import get_settings
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.constants import OPENAI_MODELS_DATA
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.test_case import MLLMImage
from deepeval.utils import (
    convert_to_multi_modal_array,
    check_if_multimodal,
    require_param,
)
from deepeval.models.llms.utils import (
    trim_and_load_json,
)
from deepeval.models.utils import (
    parse_model_name,
    require_secret_api_key,
    require_costs,
    normalize_kwargs_and_extract_aliases,
)
from deepeval.constants import ProviderSlug as PS

retry_azure = create_retry_decorator(PS.AZURE)

_ALIAS_MAP = {
    "api_key": ["azure_openai_api_key"],
    "base_url": ["azure_endpoint"],
}


class AzureOpenAIModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        azure_ad_token_provider: Optional[
            Callable[[], "str | Awaitable[str]"]
        ] = None,
        azure_ad_token: Optional[str] = None,
        temperature: Optional[float] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        deployment_name: Optional[str] = None,
        api_version: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()
        normalized_kwargs, alias_values = normalize_kwargs_and_extract_aliases(
            "AzureOpenAIModel",
            kwargs,
            _ALIAS_MAP,
        )

        # re-map deprecated keywords to re-named positional args
        if api_key is None and "api_key" in alias_values:
            api_key = alias_values["api_key"]
        if base_url is None and "base_url" in alias_values:
            base_url = alias_values["base_url"]

        # fetch Azure deployment parameters
        model = model or settings.AZURE_MODEL_NAME
        deployment_name = deployment_name or settings.AZURE_DEPLOYMENT_NAME

        self.azure_ad_token_provider = azure_ad_token_provider

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.AZURE_OPENAI_API_KEY

        if azure_ad_token is not None:
            self.azure_ad_token = azure_ad_token
        else:
            self.azure_ad_token = settings.AZURE_OPENAI_AD_TOKEN

        api_version = api_version or settings.OPENAI_API_VERSION
        if base_url is not None:
            base_url = str(base_url).rstrip("/")
        elif settings.AZURE_OPENAI_ENDPOINT is not None:
            base_url = str(settings.AZURE_OPENAI_ENDPOINT).rstrip("/")

        if temperature is not None:
            temperature = float(temperature)
        elif settings.TEMPERATURE is not None:
            temperature = settings.TEMPERATURE
        else:
            temperature = 0.0

        cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else settings.OPENAI_COST_PER_INPUT_TOKEN
        )
        cost_per_output_token = (
            cost_per_output_token
            if cost_per_output_token is not None
            else settings.OPENAI_COST_PER_OUTPUT_TOKEN
        )

        # validation
        model = require_param(
            model,
            provider_label="AzureOpenAIModel",
            env_var_name="AZURE_MODEL_NAME",
            param_hint="model",
        )

        self.deployment_name = require_param(
            deployment_name,
            provider_label="AzureOpenAIModel",
            env_var_name="AZURE_DEPLOYMENT_NAME",
            param_hint="deployment_name",
        )

        self.base_url = require_param(
            base_url,
            provider_label="AzureOpenAIModel",
            env_var_name="AZURE_OPENAI_ENDPOINT",
            param_hint="base_url",
        )

        self.api_version = require_param(
            api_version,
            provider_label="AzureOpenAIModel",
            env_var_name="OPENAI_API_VERSION",
            param_hint="api_version",
        )

        self.model_data = OPENAI_MODELS_DATA.get(model)
        cost_per_input_token, cost_per_output_token = require_costs(
            self.model_data,
            model,
            "OPENAI_COST_PER_INPUT_TOKEN",
            "OPENAI_COST_PER_OUTPUT_TOKEN",
            cost_per_input_token,
            cost_per_output_token,
        )
        self.model_data.input_price = cost_per_input_token
        self.model_data.output_price = cost_per_output_token

        if temperature < 0:
            raise DeepEvalError("Temperature must be >= 0.")
        self.temperature = temperature

        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = normalized_kwargs
        self.kwargs.pop(
            "temperature", None
        )  # to avoid duplicate with self.temperature

        self.generation_kwargs = dict(generation_kwargs or {})
        self.generation_kwargs.pop(
            "temperature", None
        )  # to avoid duplicate with self.temperature

        super().__init__(parse_model_name(model))

    ###############################################
    # Other generate functions
    ###############################################

    @retry_azure
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        client = self.load_model(async_mode=False)

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]

        if schema:
            if self.model_data.supports_structured_outputs:
                completion = client.beta.chat.completions.parse(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": content}],
                    response_format=schema,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                )
                structured_output: BaseModel = completion.choices[
                    0
                ].message.parsed
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                return structured_output, cost
            if self.model_data.supports_json:
                completion = client.beta.chat.completions.parse(
                    model=self.deployment_name,
                    messages=[
                        {"role": "user", "content": content},
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

        completion = client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {"role": "user", "content": content},
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

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]

        if schema:
            if self.model_data.supports_structured_outputs:
                completion = await client.beta.chat.completions.parse(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": content}],
                    response_format=schema,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                )
                structured_output: BaseModel = completion.choices[
                    0
                ].message.parsed
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                )
                return structured_output, cost
            if self.model_data.supports_json:
                completion = await client.beta.chat.completions.parse(
                    model=self.deployment_name,
                    messages=[
                        {"role": "user", "content": content},
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
                {"role": "user", "content": content},
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
        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]
        completion = client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": content}],
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
        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]
        completion = await client.chat.completions.create(
            model=self.deployment_name,
            messages=[{"role": "user", "content": content}],
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

    def generate_content(
        self, multimodal_input: Optional[List[Union[str, MLLMImage]]] = None
    ):
        multimodal_input = [] if multimodal_input is None else multimodal_input
        content = []
        for element in multimodal_input:
            if isinstance(element, str):
                content.append({"type": "text", "text": element})
            elif isinstance(element, MLLMImage):
                if element.url and not element.local:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": element.url},
                        }
                    )
                else:
                    element.ensure_images_loaded()
                    data_uri = (
                        f"data:{element.mimeType};base64,{element.dataBase64}"
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        }
                    )
        return content

    ###############################################
    # Utilities
    ###############################################

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        if self.model_data.input_price and self.model_data.output_price:
            input_cost = input_tokens * self.model_data.input_price
            output_cost = output_tokens * self.model_data.output_price
            return input_cost + output_cost

    ###############################################
    # Capabilities
    ###############################################

    def supports_log_probs(self) -> Union[bool, None]:
        return self.model_data.supports_log_probs

    def supports_temperature(self) -> Union[bool, None]:
        return self.model_data.supports_temperature

    def supports_multimodal(self) -> Union[bool, None]:
        return self.model_data.supports_multimodal

    def supports_structured_outputs(self) -> Union[bool, None]:
        return self.model_data.supports_structured_outputs

    def supports_json_mode(self) -> Union[bool, None]:
        return self.model_data.supports_json

    ###############################################
    # Model
    ###############################################

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
        # Only require the API key / Azure ad token if no token provider is supplied
        azure_ad_token = None
        api_key = None

        if self.azure_ad_token_provider is None:
            if self.azure_ad_token is not None:
                azure_ad_token = require_secret_api_key(
                    self.azure_ad_token,
                    provider_label="AzureOpenAI",
                    env_var_name="AZURE_OPENAI_AD_TOKEN",
                    param_hint="`azure_ad_token` to AzureOpenAIModel(...)",
                )
            else:
                api_key = require_secret_api_key(
                    self.api_key,
                    provider_label="AzureOpenAI",
                    env_var_name="AZURE_OPENAI_API_KEY",
                    param_hint="`api_key` to AzureOpenAIModel(...)",
                )

        kw = dict(
            api_key=api_key,
            api_version=self.api_version,
            azure_endpoint=self.base_url,
            azure_deployment=self.deployment_name,
            azure_ad_token_provider=self.azure_ad_token_provider,
            azure_ad_token=azure_ad_token,
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

    def get_model_name(self):
        return f"{self.name} (Azure)"
