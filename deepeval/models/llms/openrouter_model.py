import warnings
import inspect

from typing import Optional, Tuple, Union, Dict, Type
from pydantic import BaseModel, SecretStr
from openai.types.chat.chat_completion import ChatCompletion
from openai import (
    OpenAI,
    AsyncOpenAI,
)

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.errors import DeepEvalError
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.constants import DEFAULT_OPENROUTER_MODEL
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import require_secret_api_key
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)


retry_openrouter = create_retry_decorator(PS.OPENROUTER)


def _request_timeout_seconds() -> float:
    timeout = float(get_settings().DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS or 0)
    return timeout if timeout > 0 else 30.0


def _convert_schema_to_openrouter_format(
    schema: Union[Type[BaseModel], BaseModel],
) -> Dict:
    """
    Convert Pydantic BaseModel to OpenRouter's JSON Schema format.

    OpenRouter expects:
    {
        "type": "json_schema",
        "json_schema": {
            "name": "schema_name",
            "strict": true,
            "schema": { ... JSON Schema ... }
        }
    }
    """
    json_schema = schema.model_json_schema()
    schema_name = (
        schema.__name__
        if inspect.isclass(schema)
        else schema.__class__.__name__
    )

    # OpenRouter requires additionalProperties: false when strict: true
    # Ensure it's set at the root level of the schema
    if "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False

    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": json_schema,
        },
    }


class OpenRouterModel(DeepEvalBaseLLM):
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
        model = model or settings.OPENROUTER_MODEL_NAME
        if model is None:
            model = DEFAULT_OPENROUTER_MODEL

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.OPENROUTER_API_KEY

        if base_url is not None:
            base_url = str(base_url).rstrip("/")
        elif settings.OPENROUTER_BASE_URL is not None:
            base_url = str(settings.OPENROUTER_BASE_URL).rstrip("/")
        else:
            base_url = "https://openrouter.ai/api/v1"

        cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else settings.OPENROUTER_COST_PER_INPUT_TOKEN
        )
        cost_per_output_token = (
            cost_per_output_token
            if cost_per_output_token is not None
            else settings.OPENROUTER_COST_PER_OUTPUT_TOKEN
        )

        if temperature is not None:
            temperature = float(temperature)
        elif settings.TEMPERATURE is not None:
            temperature = settings.TEMPERATURE
        else:
            temperature = 0.0

        # validation
        if temperature < 0:
            raise DeepEvalError("Temperature must be >= 0.")

        self.base_url = base_url
        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self.temperature = temperature

        self.kwargs = dict(kwargs)
        self.kwargs.pop("temperature", None)

        self.generation_kwargs = dict(generation_kwargs or {})
        self.generation_kwargs.pop("temperature", None)

        super().__init__(model)

    ###############################################
    # Generate functions
    ###############################################

    async def _generate_with_client(
        self,
        client: AsyncOpenAI,
        prompt: str,
        schema: Optional[BaseModel] = None,
    ) -> Tuple[Union[str, Dict], float]:
        """
        Core generation logic shared between generate() and a_generate().

        Args:
            client: AsyncOpenAI client
            prompt: The prompt to send
            schema: Optional Pydantic schema for structured outputs

        Returns:
            Tuple of (output, cost)
        """
        if schema:
            # Try OpenRouter's native JSON Schema format
            try:
                openrouter_response_format = (
                    _convert_schema_to_openrouter_format(schema)
                )
                completion = await client.chat.completions.create(
                    model=self.name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=openrouter_response_format,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                )

                # Parse the JSON response and validate against schema
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
                # Warn if structured outputs fail
                warnings.warn(
                    f"Structured outputs not supported for model '{self.name}'. "
                    f"Falling back to regular generation with JSON parsing. "
                    f"Error: {str(e)}",
                    UserWarning,
                    stacklevel=3,
                )
                # Fall back to regular generation and parse JSON manually (like Bedrock)
                # This works with any model that can generate JSON in text
                pass

        # Regular generation (or fallback if structured outputs failed)
        completion = await client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
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
            # Parse JSON from text and validate against schema (like Bedrock)
            json_output = trim_and_load_json(output)
            return schema.model_validate(json_output), cost
        else:
            return output, cost

    @retry_openrouter
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        from deepeval.models.llms.utils import safe_asyncio_run

        client = self.load_model(async_mode=True)
        return safe_asyncio_run(
            self._generate_with_client(client, prompt, schema)
        )

    @retry_openrouter
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        client = self.load_model(async_mode=True)
        return await self._generate_with_client(client, prompt, schema)

    ###############################################
    # Other generate functions
    ###############################################

    @retry_openrouter
    def generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[ChatCompletion, float]:
        # Generate completion
        client = self.load_model(async_mode=False)
        completion = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **self.generation_kwargs,
        )
        # Cost calculation
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = self.calculate_cost(
            input_tokens, output_tokens, response=completion
        )

        return completion, cost

    @retry_openrouter
    async def a_generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[ChatCompletion, float]:
        # Generate completion
        client = self.load_model(async_mode=True)
        completion = await client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **self.generation_kwargs,
        )
        # Cost calculation
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = self.calculate_cost(
            input_tokens, output_tokens, response=completion
        )

        return completion, cost

    @retry_openrouter
    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[list[str], float]:
        client = self.load_model(async_mode=False)
        response = client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
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
    # Utilities
    ###############################################

    def calculate_cost(
        self, input_tokens: int, output_tokens: int, response=None
    ) -> Optional[float]:
        """
        Calculate cost with priority:
        1. User-provided pricing (highest priority)
        2. Try to extract from API response (if OpenRouter includes pricing)
        3. Return None if cost cannot be determined
        """
        # Priority 1: User-provided pricing
        if (
            self.cost_per_input_token is not None
            and self.cost_per_output_token is not None
        ):
            return (
                input_tokens * self.cost_per_input_token
                + output_tokens * self.cost_per_output_token
            )

        # Priority 2: Try to extract from API response (if OpenRouter includes pricing)
        # Note: OpenRouter may include pricing in response metadata
        if response is not None:
            # Check if response has cost information
            usage_cost = getattr(getattr(response, "usage", None), "cost", None)
            if usage_cost is not None:
                try:
                    return float(usage_cost)
                except (ValueError, TypeError):
                    pass
            # Some responses might have cost at the top level
            response_cost = getattr(response, "cost", None)
            if response_cost is not None:
                try:
                    return float(response_cost)
                except (ValueError, TypeError):
                    pass

        # Priority 3: Return None since cost is unknown
        return None

    ###############################################
    # Model
    ###############################################

    def get_model_name(self):
        return f"{self.name} (OpenRouter)"

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(OpenAI)
        return self._build_client(AsyncOpenAI)

    def _client_kwargs(self) -> Dict:
        """
        If Tenacity is managing retries, force OpenAI SDK retries off to avoid double retries.
        If the user opts into SDK retries for 'openrouter' via DEEPEVAL_SDK_RETRY_PROVIDERS,
        leave their retry settings as is.
        """
        kwargs = dict(self.kwargs or {})
        if not sdk_retries_for(PS.OPENROUTER):
            kwargs["max_retries"] = 0

        if not kwargs.get("timeout"):
            kwargs["timeout"] = _request_timeout_seconds()

        return kwargs

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="OpenRouter",
            env_var_name="OPENROUTER_API_KEY",
            param_hint="`api_key` to OpenRouterModel(...)",
        )

        kw = dict(
            api_key=api_key,
            base_url=self.base_url,
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
