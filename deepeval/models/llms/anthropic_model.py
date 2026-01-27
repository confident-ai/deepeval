from typing import Optional, Tuple, Union, Dict, List
from pydantic import BaseModel, SecretStr

from deepeval.errors import DeepEvalError
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.models.utils import (
    require_costs,
    require_secret_api_key,
    normalize_kwargs_and_extract_aliases,
)
from deepeval.test_case import MLLMImage
from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array
from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.utils import require_dependency, require_param
from deepeval.models.llms.constants import ANTHROPIC_MODELS_DATA

# consistent retry rules
retry_anthropic = create_retry_decorator(PS.ANTHROPIC)

_ALIAS_MAP = {
    "api_key": ["_anthropic_api_key"],
}

default_model = "claude-3-7-sonnet-latest"


class AnthropicModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()
        normalized_kwargs, alias_values = normalize_kwargs_and_extract_aliases(
            "AnthropicModel",
            kwargs,
            _ALIAS_MAP,
        )

        # re-map depricated keywords to re-named positional args
        if api_key is None and "api_key" in alias_values:
            api_key = alias_values["api_key"]

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.ANTHROPIC_API_KEY

        model = model or settings.ANTHROPIC_MODEL_NAME or default_model

        if temperature is not None:
            temperature = float(temperature)
        elif settings.TEMPERATURE is not None:
            temperature = settings.TEMPERATURE
        else:
            temperature = 0.0

        cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else settings.ANTHROPIC_COST_PER_INPUT_TOKEN
        )
        cost_per_output_token = (
            cost_per_output_token
            if cost_per_output_token is not None
            else settings.ANTHROPIC_COST_PER_OUTPUT_TOKEN
        )

        # Validation
        model = require_param(
            model,
            provider_label="AnthropicModel",
            env_var_name="ANTHROPIC_MODEL_NAME",
            param_hint="model",
        )

        if temperature < 0:
            raise DeepEvalError("Temperature must be >= 0.")
        self.temperature = temperature

        self.model_data = ANTHROPIC_MODELS_DATA.get(model)

        cost_per_input_token, cost_per_output_token = require_costs(
            self.model_data,
            model,
            "ANTHROPIC_COST_PER_INPUT_TOKEN",
            "ANTHROPIC_COST_PER_OUTPUT_TOKEN",
            cost_per_input_token,
            cost_per_output_token,
        )
        self.model_data.input_price = cost_per_input_token
        self.model_data.output_price = cost_per_output_token

        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = normalized_kwargs
        self.kwargs.pop(
            "temperature", None
        )  # to avoid duplicate with self.temperature
        max_tokens = self.kwargs.pop("max_tokens", None)

        self.generation_kwargs = dict(generation_kwargs or {})
        self.generation_kwargs.pop(
            "temperature", None
        )  # to avoid duplicate with self.temperature
        default_max_tokens = 1024 if max_tokens is None else max_tokens
        self._max_tokens = int(
            self.generation_kwargs.pop("max_tokens", default_max_tokens)
        )

        super().__init__(model)

    ###############################################
    # Generate functions
    ###############################################

    @retry_anthropic
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]

        # Get max_tokens from kwargs, default to 1024 if not provided
        max_tokens = self._max_tokens
        chat_model = self.load_model()
        message = chat_model.messages.create(
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model=self.name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        cost = self.calculate_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )
        if schema is None:
            return message.content[0].text, cost
        else:
            json_output = trim_and_load_json(message.content[0].text)
            return schema.model_validate(json_output), cost

    @retry_anthropic
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]

        # Get max_tokens from kwargs, default to 1024 if not provided
        max_tokens = self._max_tokens
        chat_model = self.load_model(async_mode=True)
        message = await chat_model.messages.create(
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model=self.name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        cost = self.calculate_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )
        if schema is None:
            return message.content[0].text, cost
        else:
            json_output = trim_and_load_json(message.content[0].text)

            return schema.model_validate(json_output), cost

    def generate_content(self, multimodal_input: List[Union[str, MLLMImage]]):
        content = []
        for element in multimodal_input:
            if isinstance(element, str):
                content.append({"type": "text", "text": element})
            elif isinstance(element, MLLMImage):
                if element.url and not element.local:
                    content.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": element.url},
                        }
                    )
                else:
                    element.ensure_images_loaded()
                    mime_type = element.mimeType or "image/jpeg"
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": element.dataBase64,
                            },
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

    #########################
    # Capabilities          #
    #########################

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
        module = require_dependency(
            "anthropic",
            provider_label="AnthropicModel",
            install_hint="Install it with `pip install anthropic`.",
        )

        if not async_mode:
            return self._build_client(module.Anthropic)
        return self._build_client(module.AsyncAnthropic)

    def _client_kwargs(self) -> Dict:
        kwargs = dict(self.kwargs or {})
        # If we are managing retries with Tenacity, force SDK retries off to avoid double retries.
        # if the user opts into SDK retries via DEEPEVAL_SDK_RETRY_PROVIDERS, then honor their max_retries.
        if not sdk_retries_for(PS.ANTHROPIC):
            kwargs["max_retries"] = 0
        return kwargs

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="Anthropic",
            env_var_name="ANTHROPIC_API_KEY",
            param_hint="`api_key` to AnthropicModel(...)",
        )
        kw = dict(
            api_key=api_key,
            **self._client_kwargs(),
        )
        try:
            return cls(**kw)
        except TypeError as e:
            # in case older SDKs don’t accept max_retries, drop it and retry
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise

    def get_model_name(self):
        return f"{self.name} (Anthropic)"
