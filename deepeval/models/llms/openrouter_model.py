from openai.types.chat.chat_completion import ChatCompletion
from deepeval.key_handler import ModelKeyValues, KEY_FILE_HANDLER
from typing import Optional, Tuple, Union, Dict, Callable, Any, Type
from pydantic import BaseModel
import warnings
import inspect

from openai import (
    OpenAI,
    AsyncOpenAI,
)

from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.models.utils import parse_model_name
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)

retry_openrouter = create_retry_decorator(PS.OPENROUTER)

# OpenRouter uses provider/model format (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
# We accept ANY model string - no validation against a hardcoded list
# This allows flexibility as OpenRouter's model catalog changes frequently

default_openrouter_model = "openai/gpt-4o-mini"

def _request_timeout_seconds() -> float:
    timeout = float(get_settings().DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS or 0)
    return timeout if timeout > 0 else 30.0


def _convert_schema_to_openrouter_format(schema: Union[Type[BaseModel], BaseModel]) -> Dict:
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
    schema_name = schema.__name__ if inspect.isclass(schema) else schema.__class__.__name__
    
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
        }
    }


class OpenRouterModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        _openrouter_api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cost_per_input_token: Optional[float] = None,
        cost_per_output_token: Optional[float] = None,
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        http_referer: Optional[str] = None,
        x_title: Optional[str] = None,
        **kwargs,
    ):
        model_name = None
        model = model or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.OPENROUTER_MODEL_NAME
        )
        cost_per_input_token = (
            cost_per_input_token
            if cost_per_input_token is not None
            else KEY_FILE_HANDLER.fetch_data(
                ModelKeyValues.OPENROUTER_COST_PER_INPUT_TOKEN
            )
        )
        cost_per_output_token = (
            cost_per_output_token
            if cost_per_output_token is not None
            else KEY_FILE_HANDLER.fetch_data(
                ModelKeyValues.OPENROUTER_COST_PER_OUTPUT_TOKEN
            )
        )

        if isinstance(model, str):
            model_name = parse_model_name(model)
        elif model is None:
            model_name = default_openrouter_model

        # Store user-provided pricing (if given) - highest priority
        self.cost_per_input_token = (
            float(cost_per_input_token) if cost_per_input_token is not None else None
        )
        self.cost_per_output_token = (
            float(cost_per_output_token) if cost_per_output_token is not None else None
        )

        self._openrouter_api_key = _openrouter_api_key
        # Default to OpenRouter's API endpoint if not provided
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        self.http_referer = http_referer
        self.x_title = x_title

        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_name)

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
                openrouter_response_format = _convert_schema_to_openrouter_format(schema)
                completion = await client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=openrouter_response_format,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                )
                
                # Parse the JSON response and validate against schema
                json_output = trim_and_load_json(completion.choices[0].message.content)
                cost = self.calculate_cost(
                    completion.usage.prompt_tokens,
                    completion.usage.completion_tokens,
                    response=completion,
                )
                return schema.model_validate(json_output), cost
            except Exception as e:
                # Warn if structured outputs fail
                warnings.warn(
                    f"Structured outputs not supported for model '{self.model_name}'. "
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
            model=self.model_name,
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
        return safe_asyncio_run(self._generate_with_client(client, prompt, schema))

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
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **self.generation_kwargs,
        )
        # Cost calculation
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = self.calculate_cost(input_tokens, output_tokens, response=completion)

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
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=top_logprobs,
            **self.generation_kwargs,
        )
        # Cost calculation
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        cost = self.calculate_cost(input_tokens, output_tokens, response=completion)

        return completion, cost

    @retry_openrouter
    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[list[str], float]:
        client = self.load_model(async_mode=False)
        response = client.chat.completions.create(
            model=self.model_name,
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
    ) -> float:
        """
        Calculate cost with priority:
        1. User-provided pricing (highest priority)
        2. Try to extract from API response (if OpenRouter includes pricing)
        3. Return 0 (cost tracking disabled for unknown models)
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

        # Priority 3: Return 0 (cost tracking disabled for unknown models)
        # This allows the model to work even without pricing information
        return 0.0

    ###############################################
    # Model
    ###############################################

    def get_model_name(self):
        return self.model_name

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

        # Add OpenRouter-specific headers
        default_headers = kwargs.get("default_headers", {})
        if self.http_referer:
            default_headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            default_headers["X-Title"] = self.x_title
        if default_headers:
            kwargs["default_headers"] = default_headers

        return kwargs

    def _build_client(self, cls):
        settings = get_settings()
        api_key = (
            settings.OPENROUTER_API_KEY.get_secret_value() 
            if settings.OPENROUTER_API_KEY is not None 
            else None
        ) or self._openrouter_api_key
        
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