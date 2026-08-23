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
    EvaluationCost,
)
from deepeval.test_case import MLLMImage
from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array
from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.utils import require_dependency, require_param
from deepeval.models.llms.constants import (
    ANTHROPIC_MODELS_DATA,
    DEFAULT_ANTHROPIC_MODEL,
)

# consistent retry rules
retry_anthropic = create_retry_decorator(PS.ANTHROPIC)

# Anthropic's `max_tokens` caps thinking *plus* response text, and its minimum
# thinking budget is 1024, so a thinking request needs headroom for both.
MIN_THINKING_BUDGET_TOKENS = 1024
DEFAULT_MAX_TOKENS = 1024
DEFAULT_THINKING_MAX_TOKENS = 8192

_ALIAS_MAP = {
    "api_key": ["_anthropic_api_key"],
}


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

        model = (
            model or settings.ANTHROPIC_MODEL_NAME or DEFAULT_ANTHROPIC_MODEL
        )

        if temperature is not None:
            temperature = float(temperature)
        elif settings.TEMPERATURE is not None:
            temperature = settings.TEMPERATURE
        # else: leave as None so `temperature` is only sent to the client when
        # explicitly configured — some models (e.g. reasoning models) reject it.

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

        if temperature is not None and temperature < 0:
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
        self._thinking = (
            settings.DEEPEVAL_MODEL_THINKING is True
            and self.model_data.supports_thinking is True
        )
        explicit_max_tokens = self.generation_kwargs.pop(
            "max_tokens", max_tokens
        )
        if explicit_max_tokens is not None:
            self._max_tokens = int(explicit_max_tokens)
        elif self._thinking:
            self._max_tokens = DEFAULT_THINKING_MAX_TOKENS
        else:
            self._max_tokens = DEFAULT_MAX_TOKENS

        if self._thinking:
            self._thinking_budget_tokens = max(
                MIN_THINKING_BUDGET_TOKENS, self._max_tokens // 2
            )
            if self._max_tokens <= self._thinking_budget_tokens:
                raise DeepEvalError(
                    f"Thinking needs at least "
                    f"{MIN_THINKING_BUDGET_TOKENS} tokens of budget on top of "
                    f"the response itself, but max_tokens is "
                    f"{self._max_tokens} and caps thinking and response "
                    f"together. Raise max_tokens above "
                    f"{MIN_THINKING_BUDGET_TOKENS * 2} or unset "
                    f"DEEPEVAL_MODEL_THINKING."
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

        max_tokens = self._max_tokens
        chat_model = self.load_model()
        create_kwargs = dict(
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model=self.name,
            **self.generation_kwargs,
        )
        create_kwargs.update(self._thinking_kwargs())
        # Only send `temperature` when explicitly configured and the model
        # supports it — some models reject/deprecate `temperature`, and a
        # thinking request only accepts the default.
        if not self._thinking and (
            self.temperature is not None
            and not (
                self.model_data
                and self.model_data.supports_temperature is False
            )
        ):
            create_kwargs["temperature"] = self.temperature
        message = chat_model.messages.create(**create_kwargs)
        cost = self.calculate_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )
        text = self._extract_text(message)
        if schema is None:
            return text, cost
        else:
            json_output = trim_and_load_json(text)
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

        max_tokens = self._max_tokens
        chat_model = self.load_model(async_mode=True)
        create_kwargs = dict(
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model=self.name,
            **self.generation_kwargs,
        )
        create_kwargs.update(self._thinking_kwargs())
        # Only send `temperature` when explicitly configured and the model
        # supports it — some models reject/deprecate `temperature`, and a
        # thinking request only accepts the default.
        if not self._thinking and (
            self.temperature is not None
            and not (
                self.model_data
                and self.model_data.supports_temperature is False
            )
        ):
            create_kwargs["temperature"] = self.temperature
        message = await chat_model.messages.create(**create_kwargs)
        cost = self.calculate_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )
        text = self._extract_text(message)
        if schema is None:
            return text, cost
        else:
            json_output = trim_and_load_json(text)

            return schema.model_validate(json_output), cost

    @staticmethod
    def _extract_text(message) -> str:
        """The response text, skipping any leading thinking block."""
        for block in getattr(message, "content", None) or []:
            if getattr(block, "type", None) == "text":
                return block.text
        raise DeepEvalError(
            "Anthropic returned no text block. With thinking enabled the "
            "whole token budget can go to reasoning — raise max_tokens or "
            "unset DEEPEVAL_MODEL_THINKING."
        )

    def _thinking_kwargs(self) -> Dict:
        """The `thinking` block to send, empty when it is not ours to set.

        Models that always think reject a disabled block and older ones reject
        the parameter outright, so only models the registry marks as
        `supports_thinking` get one. An explicit `thinking` in
        `generation_kwargs` is already on the request and wins.
        """
        if (
            self.model_data.supports_thinking is not True
            or "thinking" in self.generation_kwargs
        ):
            return {}
        if not self._thinking:
            return {"thinking": {"type": "disabled"}}
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": self._thinking_budget_tokens,
            }
        }

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
            return EvaluationCost(
                input_cost + output_cost, input_tokens, output_tokens
            )

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
