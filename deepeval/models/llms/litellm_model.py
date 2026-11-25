from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, SecretStr

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.errors import DeepEvalError
from deepeval.models.llms.gateway_model import (
    DeepEvalBaseGatewayModel,
    _SCHEMA_HINTS,
    _is_bad_request,
    _is_rejected_param,
)
from deepeval.models.llms.multimodal import to_openai_content
from deepeval.test_case import Audio
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import (
    EvaluationCost,
    infer_provider,
    normalize_kwargs_and_extract_aliases,
    require_secret_api_key,
)
from deepeval.utils import require_dependency, require_param

_ALIAS_MAP = {
    "base_url": ["api_base"],
}

# Which deepeval setting holds the key for each provider LiteLLM routes to.
_PROVIDER_API_KEY_SETTINGS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "xai": "GROK_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
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
        system_prompt: Optional[str] = None,
        supports_log_probs: Optional[bool] = None,
        **kwargs,
    ):
        """
        Any extra keyword argument (`reasoning_effort`, `api_version`,
        `aws_region_name`, `vertex_project`, `extra_headers`, ...) is forwarded
        as given to every LiteLLM completion call.

        `supports_log_probs` overrides what LiteLLM reports for the model:
        `True` always requests log probabilities, `False` never does.
        """
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

        if api_key is not None:
            # keep it secret, keep it safe from serializing, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = self._resolve_api_key(settings, model)

        if temperature is not None:
            temperature = float(temperature)
        elif settings.TEMPERATURE is not None:
            temperature = settings.TEMPERATURE
        # else: leave as None so `temperature` is only sent to the client when
        # explicitly configured — some models (e.g. reasoning models) reject it.

        model = require_param(
            model,
            provider_label="LiteLLMModel",
            env_var_name="LITELLM_MODEL_NAME",
            param_hint="model",
        )

        if temperature is not None and temperature < 0:
            raise DeepEvalError("Temperature must be >= 0.")
        self.temperature = temperature

        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token

        # Keep sanitized kwargs (legacy keys stripped) for the litellm call
        self.kwargs = normalized_kwargs
        self.kwargs.pop("temperature", None)

        self.generation_kwargs = dict(generation_kwargs or {})
        self.generation_kwargs.pop("temperature", None)

        self.system_prompt = system_prompt
        self._log_probs_supported = supports_log_probs
        self._structured_outputs_supported: Optional[bool] = None

        super().__init__(model)

    ###############################################
    # Generate
    ###############################################

    def _generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        audio: Optional[List[Audio]] = None,
    ) -> Tuple[Union[str, BaseModel], Optional[float]]:
        completion = self._litellm().completion
        params = self._completion_params(to_openai_content(prompt, audio))
        if schema and self._structured_outputs_supported is not False:
            try:
                response = completion(**params, response_format=schema)
            except Exception as e:
                self._note_schema_rejection(e)
                response = completion(**params)
        else:
            response = completion(**params)
        return self._parse_response(response, schema)

    async def _a_generate(
        self,
        prompt: str,
        schema: Optional[BaseModel] = None,
        audio: Optional[List[Audio]] = None,
    ) -> Tuple[Union[str, BaseModel], Optional[float]]:
        acompletion = self._litellm().acompletion
        params = self._completion_params(to_openai_content(prompt, audio))
        if schema and self._structured_outputs_supported is not False:
            try:
                response = await acompletion(**params, response_format=schema)
            except Exception as e:
                self._note_schema_rejection(e)
                response = await acompletion(**params)
        else:
            response = await acompletion(**params)
        return self._parse_response(response, schema)

    def _note_schema_rejection(self, error: Exception) -> None:
        """Re-raise unless the provider refused the request, in which case the
        caller retries without `response_format` and parses the JSON itself."""
        if not _is_bad_request(error):
            raise error
        if _is_rejected_param(error, _SCHEMA_HINTS):
            self._structured_outputs_supported = False

    ###############################################
    # Raw response + samples
    ###############################################

    def _generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
        audio: Optional[List[Audio]] = None,
    ) -> Tuple[Any, Optional[float]]:
        params = self._completion_params(to_openai_content(prompt, audio))
        params.update({"logprobs": True, "top_logprobs": top_logprobs})
        response = self._litellm().completion(**params)
        return response, self._response_cost(response)

    async def _a_generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
        audio: Optional[List[Audio]] = None,
    ) -> Tuple[Any, Optional[float]]:
        params = self._completion_params(to_openai_content(prompt, audio))
        params.update({"logprobs": True, "top_logprobs": top_logprobs})
        response = await self._litellm().acompletion(**params)
        return response, self._response_cost(response)

    def generate_samples(
        self,
        prompt: str,
        n: int,
        temperature: float,
        audio: Optional[List[Audio]] = None,
    ) -> Tuple[List[str], Optional[float]]:
        return self._run(self._generate_samples, prompt, n, temperature, audio)

    def _generate_samples(
        self,
        prompt: str,
        n: int,
        temperature: float,
        audio: Optional[List[Audio]] = None,
    ) -> Tuple[List[str], Optional[float]]:
        params = self._completion_params(to_openai_content(prompt, audio))
        params.update({"n": n, "temperature": temperature})
        response = self._litellm().completion(**params)
        samples = [choice.message.content for choice in response.choices]
        return samples, self._response_cost(response)

    ###############################################
    # Helpers
    ###############################################

    @staticmethod
    def _litellm():
        return require_dependency(
            "litellm",
            provider_label="LiteLLM",
            install_hint="Install it with `pip install litellm`.",
        )

    def _resolve_api_key(self, settings, model: Optional[str]):
        explicit = settings.LITELLM_API_KEY or settings.LITELLM_PROXY_API_KEY
        if explicit:
            return explicit
        if self.base_url:
            # A custom endpoint (e.g. a LiteLLM proxy) takes whichever key is
            # configured; the provider cannot be told from the model name.
            return (
                settings.OPENAI_API_KEY
                or settings.ANTHROPIC_API_KEY
                or settings.GOOGLE_API_KEY
            )
        # Otherwise only the key of the provider being called is sent. Unknown
        # providers get none and LiteLLM reads its own environment variables.
        setting = _PROVIDER_API_KEY_SETTINGS.get(infer_provider(model))
        return getattr(settings, setting, None) if setting else None

    def _completion_params(self, content: List[Dict]) -> Dict[str, Any]:
        messages = [{"role": "user", "content": content}]
        if self.system_prompt:
            messages.insert(
                0, {"role": "system", "content": self.system_prompt}
            )
        params: Dict[str, Any] = {
            "model": self.name,
            "messages": messages,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
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

    def _estimate_response_cost(self, response: Any) -> Optional[float]:
        """Price the response from LiteLLM's model price map."""
        try:
            cost = (getattr(response, "_hidden_params", None) or {}).get(
                "response_cost"
            )
            if cost is None:
                cost = self._litellm().completion_cost(
                    completion_response=response
                )
        except Exception:
            return None
        if isinstance(cost, (int, float)) and cost >= 0:
            return float(cost)
        return None

    def supports_log_probs(self) -> Optional[bool]:
        if self._log_probs_supported is None:
            self._log_probs_supported = self._lookup_log_probs_support()
        return self._log_probs_supported

    def _lookup_log_probs_support(self) -> Optional[bool]:
        try:
            litellm = self._litellm()
            model, provider, _, _ = litellm.get_llm_provider(
                self.name, api_base=self.base_url
            )
            params = litellm.get_supported_openai_params(
                model=model, custom_llm_provider=provider
            )
        except Exception:
            return None
        if not params:
            return None
        # LiteLLM lists a parameter only when it is known to work; a missing
        # one is treated as unknown so the request is still attempted once.
        return True if "logprobs" in params else None

    def supports_multimodal(self) -> bool:
        return True

    def get_model_name(self) -> str:
        from litellm import get_llm_provider
        from litellm.types.router import GenericLiteLLMParams

        provider = get_llm_provider(
            self.name,
            api_base=self.base_url,
            litellm_params=GenericLiteLLMParams(**self.kwargs),
        )
        return f"{self.name} ({provider})"

    def load_model(self, async_mode: bool = False):
        # litellm creates its client internally per call; nothing to load here.
        return None
