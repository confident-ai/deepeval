from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, SecretStr

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.errors import DeepEvalError
from deepeval.models.llms.gateway_model import DeepEvalBaseGatewayModel
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import (
    EvaluationCost,
    normalize_kwargs_and_extract_aliases,
    require_secret_api_key,
)
from deepeval.test_case import MLLMImage
from deepeval.utils import (
    check_if_multimodal,
    convert_to_multi_modal_array,
    require_param,
)

_ALIAS_MAP = {
    "base_url": ["api_base"],
}


class LiteLLMModel(DeepEvalBaseGatewayModel):
    """LiteLLM gateway, reached through the ``litellm`` library.

    LiteLLM is itself a meta-router, so unlike OpenRouter/Portkey it does not
    speak a single OpenAI-compatible HTTP endpoint — it dispatches through the
    ``litellm`` Python package. It therefore extends ``DeepEvalBaseGatewayModel``
    directly (sharing the retry policy, the ``(output, cost)`` contract and
    ``EvaluationCost`` accounting) while providing its own ``litellm`` transport.
    """

    PROVIDER_SLUG = PS.LITELLM
    PROVIDER_LABEL = "LiteLLM"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()
        normalized_kwargs, alias_values = normalize_kwargs_and_extract_aliases(
            "LiteLLMModel",
            kwargs,
            _ALIAS_MAP,
        )

        # re-map deprecated keyword to the renamed positional arg
        if base_url is None and "base_url" in alias_values:
            base_url = alias_values["base_url"]

        model = model or settings.LITELLM_MODEL_NAME

        if api_key is not None:
            # keep it secret, keep it safe from serializing, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = (
                settings.LITELLM_API_KEY
                or settings.LITELLM_PROXY_API_KEY
                or settings.OPENAI_API_KEY
                or settings.ANTHROPIC_API_KEY
                or settings.GOOGLE_API_KEY
            )

        base_url = (
            base_url
            or (
                str(settings.LITELLM_API_BASE)
                if settings.LITELLM_API_BASE is not None
                else None
            )
            or (
                str(settings.LITELLM_PROXY_API_BASE)
                if settings.LITELLM_PROXY_API_BASE is not None
                else None
            )
        )
        self.base_url = (
            str(base_url).rstrip("/") if base_url is not None else None
        )

        if temperature is not None:
            temperature = float(temperature)
        elif settings.TEMPERATURE is not None:
            temperature = settings.TEMPERATURE
        else:
            temperature = 0.0

        model = require_param(
            model,
            provider_label="LiteLLMModel",
            env_var_name="LITELLM_MODEL_NAME",
            param_hint="model",
        )

        if temperature < 0:
            raise DeepEvalError("Temperature must be >= 0.")
        self.temperature = temperature

        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token

        # Keep sanitized kwargs (legacy keys stripped) for the litellm call
        self.kwargs = normalized_kwargs
        self.kwargs.pop("temperature", None)

        self.generation_kwargs = dict(generation_kwargs or {})
        self.generation_kwargs.pop("temperature", None)

        super().__init__(model)

    ###############################################
    # Generate
    ###############################################

    def _generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], Optional[float]]:
        from litellm import completion

        params = self._completion_params(self._build_content(prompt))
        if schema:
            params["response_format"] = schema
        response = completion(**params)
        return self._parse_response(response, schema)

    async def _a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], Optional[float]]:
        from litellm import acompletion

        params = self._completion_params(self._build_content(prompt))
        if schema:
            params["response_format"] = schema
        response = await acompletion(**params)
        return self._parse_response(response, schema)

    ###############################################
    # Raw response + samples
    ###############################################

    def generate_raw_response(
        self, prompt: str, top_logprobs: int = 5
    ) -> Tuple[Any, Optional[float]]:
        return self._run(self._generate_raw_response, prompt, top_logprobs)

    async def a_generate_raw_response(
        self, prompt: str, top_logprobs: int = 5
    ) -> Tuple[Any, Optional[float]]:
        return await self._arun(
            self._a_generate_raw_response, prompt, top_logprobs
        )

    def _generate_raw_response(
        self, prompt: str, top_logprobs: int = 5
    ) -> Tuple[Any, Optional[float]]:
        from litellm import completion

        params = self._completion_params(self._build_content(prompt))
        params.update({"logprobs": True, "top_logprobs": top_logprobs})
        response = completion(**params)
        return response, self._response_cost(response)

    async def _a_generate_raw_response(
        self, prompt: str, top_logprobs: int = 5
    ) -> Tuple[Any, Optional[float]]:
        from litellm import acompletion

        params = self._completion_params(self._build_content(prompt))
        params.update({"logprobs": True, "top_logprobs": top_logprobs})
        response = await acompletion(**params)
        return response, self._response_cost(response)

    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[List[str], Optional[float]]:
        return self._run(self._generate_samples, prompt, n, temperature)

    def _generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[List[str], Optional[float]]:
        from litellm import completion

        params = self._completion_params(self._build_content(prompt))
        params.update({"n": n, "temperature": temperature})
        response = completion(**params)
        samples = [choice.message.content for choice in response.choices]
        return samples, self._response_cost(response)

    ###############################################
    # Helpers
    ###############################################

    def _completion_params(self, content: List[Dict]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.name,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
        }
        if self.api_key:
            params["api_key"] = require_secret_api_key(
                self.api_key,
                provider_label="LiteLLM",
                env_var_name=(
                    "LITELLM_API_KEY|LITELLM_PROXY_API_KEY|OPENAI_API_KEY|"
                    "ANTHROPIC_API_KEY|GOOGLE_API_KEY"
                ),
                param_hint="`api_key` to LiteLLMModel(...)",
            )
        if self.base_url:
            params["api_base"] = self.base_url
        params.update(self.kwargs)
        params.update(self.generation_kwargs)
        return params

    def _build_content(self, prompt: str) -> List[Dict]:
        if check_if_multimodal(prompt):
            return self.generate_content(convert_to_multi_modal_array(prompt))
        return [{"type": "text", "text": prompt}]

    def _parse_response(
        self, response: Any, schema: Optional[BaseModel]
    ) -> Tuple[Union[str, BaseModel], Optional[float]]:
        content = response.choices[0].message.content
        cost = self._response_cost(response)
        if schema:
            return schema(**trim_and_load_json(content)), cost
        return content, cost

    def _response_cost(self, response: Any) -> Optional[EvaluationCost]:
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        return self.calculate_cost(
            input_tokens, output_tokens, response=response
        )

    def generate_content(
        self, multimodal_input: Optional[List[Union[str, MLLMImage]]] = None
    ) -> List[Dict]:
        content: List[Dict] = []
        for element in multimodal_input or []:
            if isinstance(element, str):
                content.append({"type": "text", "text": element})
            elif isinstance(element, MLLMImage):
                if element.url and not element.local:
                    url = element.url
                else:
                    element.ensure_images_loaded()
                    url = f"data:{element.mimeType};base64,{element.dataBase64}"
                content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    def supports_multimodal(self) -> bool:
        return True

    def get_model_name(self) -> str:
        from litellm import get_llm_provider

        provider = get_llm_provider(self.name)
        return f"{self.name} ({provider})"

    def load_model(self, async_mode: bool = False):
        # litellm creates its client internally per call; nothing to load here.
        return None
