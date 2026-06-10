"""Shared base classes for LLM *gateway* providers.

A "gateway" (a.k.a. router / proxy) exposes one unified API that forwards a
user-chosen upstream model to many providers — e.g. OpenRouter, Portkey and
LiteLLM. Unlike the native providers (OpenAI, Anthropic, Gemini, ...), gateways
have no curated per-model capability/pricing table, so:

- cost is resolved from user-supplied pricing first, then from any cost the
  gateway reports back in its response, otherwise it is unknown (``None``);
- capability flags default to "unknown" (``None``) — the concrete upstream model
  decides what is actually supported.

Two layers live here:

``DeepEvalBaseGatewayModel``
    Transport-agnostic contract shared by *all* gateways: the centralized retry
    policy, the standard ``(output, cost)`` return contract, ``EvaluationCost``
    cost accounting and ``get_model_name``.

``DeepEvalOpenAICompatibleModel``
    Adds the concrete OpenAI-Chat-Completions transport (the ``openai`` SDK
    pointed at the gateway's ``base_url``), structured-output handling and the
    ``generate_raw_response`` / ``generate_samples`` helpers. ``OpenRouterModel``
    and ``PortkeyModel`` extend this — they only differ in configuration
    (settings keys, default base URL, auth headers).
"""

import inspect
import warnings
from typing import Dict, List, Optional, Tuple, Type, Union

from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel, SecretStr

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import safe_asyncio_run, trim_and_load_json
from deepeval.models.retry_policy import create_retry_decorator, sdk_retries_for
from deepeval.models.utils import EvaluationCost, require_secret_api_key
from deepeval.test_case import MLLMImage
from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array


def _request_timeout_seconds() -> float:
    timeout = float(get_settings().DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS or 0)
    return timeout if timeout > 0 else 30.0


class DeepEvalBaseGatewayModel(DeepEvalBaseLLM):
    """Transport-agnostic base shared by every gateway model.

    Subclasses must:
      - set ``PROVIDER_SLUG`` and ``PROVIDER_LABEL`` (used for retry wiring,
        cost lookup and the human-readable model name);
      - implement ``load_model`` (inherited abstract from ``DeepEvalBaseLLM``);
      - implement ``_generate`` / ``_a_generate``, returning ``(output, cost)``.

    The public ``generate`` / ``a_generate`` wrap those implementations with the
    centralized retry decorator so retries are consistent across all gateways.
    """

    PROVIDER_SLUG: ProviderSlug
    PROVIDER_LABEL: str = "Gateway"
    cost_per_input_token: Optional[float] = None
    cost_per_output_token: Optional[float] = None

    def __init__(self, model: Optional[str] = None, *args, **kwargs):
        # Build the retry decorator once per instance from the provider slug.
        # `dynamic_*` components inside read settings at call time, so a single
        # decorator honours runtime config changes.
        self._retry = create_retry_decorator(self.PROVIDER_SLUG)
        super().__init__(model, *args, **kwargs)

    ###############################################
    # Retry helpers
    ###############################################

    def _run(self, fn, *args, **kwargs):
        """Run a sync implementation under the centralized retry policy."""
        return self._retry(fn)(*args, **kwargs)

    async def _arun(self, fn, *args, **kwargs):
        """Run an async implementation under the centralized retry policy."""
        return await self._retry(fn)(*args, **kwargs)

    ###############################################
    # Public generate contract
    ###############################################

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], Optional[float]]:
        return self._run(self._generate, prompt, schema)

    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], Optional[float]]:
        return await self._arun(self._a_generate, prompt, schema)

    def _generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], Optional[float]]:
        raise NotImplementedError

    async def _a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], Optional[float]]:
        raise NotImplementedError

    ###############################################
    # Cost
    ###############################################

    def calculate_cost(
        self,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        response=None,
    ) -> Optional[EvaluationCost]:
        if (
            self.cost_per_input_token is not None
            and self.cost_per_output_token is not None
        ):
            in_tokens = input_tokens or 0
            out_tokens = output_tokens or 0
            return EvaluationCost(
                in_tokens * self.cost_per_input_token
                + out_tokens * self.cost_per_output_token,
                input_tokens,
                output_tokens,
            )

        reported = self._extract_response_cost(response)
        if reported is not None:
            return EvaluationCost(reported, input_tokens, output_tokens)

        return None

    @staticmethod
    def _extract_response_cost(response) -> Optional[float]:
        if response is None:
            return None
        candidates = (
            getattr(getattr(response, "usage", None), "cost", None),
            getattr(response, "cost", None),
        )
        for candidate in candidates:
            if candidate is not None:
                try:
                    return float(candidate)
                except (ValueError, TypeError):
                    continue
        return None

    ###############################################
    # Model identity
    ###############################################

    def get_model_name(self) -> str:
        return f"{self.name} ({self.PROVIDER_LABEL})"


class DeepEvalOpenAICompatibleModel(DeepEvalBaseGatewayModel):
    """Base for gateways reachable through the OpenAI Chat Completions API.

    Concrete subclasses (OpenRouter, Portkey) parse their own configuration in
    ``__init__`` and set the following instance attributes before calling
    ``super().__init__(model)``:

      - ``self.api_key``: ``SecretStr`` | ``None``
      - ``self.base_url``: ``str``
      - ``self.temperature``: ``float``
      - ``self.generation_kwargs``: ``dict`` (forwarded to ``create(...)``)
      - ``self.kwargs``: ``dict`` (forwarded to the OpenAI client constructor)

    They may also override ``API_KEY_ENV_VAR`` / ``API_KEY_PARAM_HINT`` (used in
    the missing-key error message) and ``_client_extra_kwargs`` (e.g. to inject
    provider-specific default headers).
    """

    API_KEY_ENV_VAR: str = "API_KEY"
    API_KEY_PARAM_HINT: str = "api_key"

    ###############################################
    # Generate (structured-output aware)
    ###############################################

    def _generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], Optional[float]]:
        # Reuse the async path so structured-output handling lives in one place.
        return safe_asyncio_run(self._a_generate(prompt, schema))

    async def _a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], Optional[float]]:
        client = self.load_model(async_mode=True)
        messages = self._build_messages(prompt)

        if schema:
            # Try the gateway's native JSON-Schema structured output first.
            try:
                completion = await client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    response_format=self._schema_response_format(schema),
                    temperature=self.temperature,
                    **self.generation_kwargs,
                )
                json_output = trim_and_load_json(
                    completion.choices[0].message.content
                )
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                    response=completion,
                )
                return schema.model_validate(json_output), cost
            except Exception as e:
                warnings.warn(
                    f"Structured outputs not supported for model '{self.name}'. "
                    f"Falling back to regular generation with JSON parsing. "
                    f"Error: {str(e)}",
                    UserWarning,
                    stacklevel=3,
                )

        completion = await client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        output = completion.choices[0].message.content
        cost = self.calculate_cost(
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
            response=completion,
        )
        if schema:
            return schema.model_validate(trim_and_load_json(output)), cost
        return output, cost

    ###############################################
    # Raw response + samples (logprob-based metrics)
    ###############################################

    def generate_raw_response(
        self, prompt: str, top_logprobs: int = 5
    ) -> Tuple[object, Optional[float]]:
        return self._run(self._generate_raw_response, prompt, top_logprobs)

    async def a_generate_raw_response(
        self, prompt: str, top_logprobs: int = 5
    ) -> Tuple[object, Optional[float]]:
        return await self._arun(
            self._a_generate_raw_response, prompt, top_logprobs
        )

    def _generate_raw_response(
        self, prompt: str, top_logprobs: int = 5
    ) -> Tuple[object, Optional[float]]:
        client = self.load_model(async_mode=False)
        completion = client.chat.completions.create(
            model=self.name,
            messages=self._build_messages(prompt),
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **self.generation_kwargs,
        )
        cost = self.calculate_cost(
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
            response=completion,
        )
        return completion, cost

    async def _a_generate_raw_response(
        self, prompt: str, top_logprobs: int = 5
    ) -> Tuple[object, Optional[float]]:
        client = self.load_model(async_mode=True)
        completion = await client.chat.completions.create(
            model=self.name,
            messages=self._build_messages(prompt),
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **self.generation_kwargs,
        )
        cost = self.calculate_cost(
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
            response=completion,
        )
        return completion, cost

    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[List[str], Optional[float]]:
        return self._run(self._generate_samples, prompt, n, temperature)

    def _generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[List[str], Optional[float]]:
        client = self.load_model(async_mode=False)
        response = client.chat.completions.create(
            model=self.name,
            messages=self._build_messages(prompt),
            n=n,
            temperature=temperature,
            **self.generation_kwargs,
        )
        completions = [choice.message.content for choice in response.choices]
        cost = self.calculate_cost(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response=response,
        )
        return completions, cost

    ###############################################
    # Capabilities
    ###############################################

    def supports_multimodal(self) -> bool:
        # The OpenAI content-array format we emit carries images, so multimodal
        # is available whenever the chosen upstream model supports it.
        return True

    ###############################################
    # Message + schema helpers
    ###############################################

    def _build_messages(self, prompt: str) -> List[Dict]:
        if check_if_multimodal(prompt):
            content = self.generate_content(
                convert_to_multi_modal_array(input=prompt)
            )
        else:
            content = prompt
        return [{"role": "user", "content": content}]

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

    @staticmethod
    def _schema_response_format(
        schema: Union[Type[BaseModel], BaseModel],
    ) -> Dict:
        json_schema = schema.model_json_schema()
        schema_name = (
            schema.__name__
            if inspect.isclass(schema)
            else schema.__class__.__name__
        )
        # `strict: true` requires additionalProperties to be set at the root.
        json_schema.setdefault("additionalProperties", False)
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": json_schema,
            },
        }

    ###############################################
    # OpenAI client construction
    ###############################################

    def load_model(self, async_mode: bool = False):
        return self._build_client(AsyncOpenAI if async_mode else OpenAI)

    def _client_kwargs(self) -> Dict:
        kwargs = dict(self.kwargs or {})
        if not sdk_retries_for(self.PROVIDER_SLUG):
            kwargs["max_retries"] = 0
        if not kwargs.get("timeout"):
            kwargs["timeout"] = _request_timeout_seconds()
        return kwargs

    def _client_extra_kwargs(self) -> Dict:
        """Hook for provider-specific client kwargs (e.g. default headers)."""
        return {}

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label=self.PROVIDER_LABEL,
            env_var_name=self.API_KEY_ENV_VAR,
            param_hint=self.API_KEY_PARAM_HINT,
        )

        kw = dict(
            api_key=api_key,
            base_url=self.base_url,
            **self._client_kwargs(),
        )
        kw.update(self._client_extra_kwargs())
        try:
            return cls(**kw)
        except TypeError as e:
            # Older OpenAI SDKs may not accept max_retries; drop it and retry once.
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise
