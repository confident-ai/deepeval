import logging
from typing import Optional, Tuple, Union, Dict, List, Any
from pydantic import BaseModel, SecretStr
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    wait_exponential_jitter,
    RetryCallState,
)

from deepeval.errors import DeepEvalError
from deepeval.config.settings import get_settings
from deepeval.models.utils import (
    require_secret_api_key,
    normalize_kwargs_and_extract_aliases,
)
from deepeval.test_case import MLLMImage
from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.utils import require_param


def log_retry_error(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    logging.error(
        f"LiteLLM Error: {exception} Retrying: {retry_state.attempt_number} time(s)..."
    )


# Define retryable exceptions
retryable_exceptions = (
    Exception,  # LiteLLM handles specific exceptions internally
)

_ALIAS_MAP = {
    "base_url": ["api_base"],
}


class LiteLLMModel(DeepEvalBaseLLM):
    EXP_BASE: int = 2
    INITIAL_WAIT: int = 1
    JITTER: int = 2
    MAX_RETRIES: int = 6
    MAX_WAIT: int = 10

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()
        normalized_kwargs, alias_values = normalize_kwargs_and_extract_aliases(
            "LiteLLMModel",
            kwargs,
            _ALIAS_MAP,
        )

        # re-map depricated keywords to re-named positional args
        if base_url is None and "base_url" in alias_values:
            base_url = alias_values["base_url"]

        # Get model name from parameter or key file
        model = model or settings.LITELLM_MODEL_NAME

        # Get API key from parameter, or settings
        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and aolike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = (
                settings.LITELLM_API_KEY
                or settings.LITELLM_PROXY_API_KEY
                or settings.OPENAI_API_KEY
                or settings.ANTHROPIC_API_KEY
                or settings.GOOGLE_API_KEY
            )

        # Get API base from parameter, key file, or environment variable
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

        # validation
        model = require_param(
            model,
            provider_label="LiteLLMModel",
            env_var_name="LITELLM_MODEL_NAME",
            param_hint="model",
        )

        if temperature < 0:
            raise DeepEvalError("Temperature must be >= 0.")
        self.temperature = temperature
        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = normalized_kwargs
        self.kwargs.pop("temperature", None)

        self.generation_kwargs = dict(generation_kwargs or {})
        self.generation_kwargs.pop("temperature", None)

        self.evaluation_cost = 0.0  # Initialize cost to 0.0
        super().__init__(model)

    @retry(
        wait=wait_exponential_jitter(
            initial=INITIAL_WAIT, exp_base=EXP_BASE, jitter=JITTER, max=MAX_WAIT
        ),
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:

        from litellm import completion

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]

        completion_params = {
            "model": self.name,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
        }

        if self.api_key:
            api_key = require_secret_api_key(
                self.api_key,
                provider_label="LiteLLM",
                env_var_name="LITELLM_API_KEY|LITELLM_PROXY_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY",
                param_hint="`api_key` to LiteLLMModel(...)",
            )
            completion_params["api_key"] = api_key
        if self.base_url:
            completion_params["api_base"] = self.base_url

        # Add schema if provided
        if schema:
            completion_params["response_format"] = schema

        # Add any additional parameters
        completion_params.update(self.kwargs)
        completion_params.update(self.generation_kwargs)

        try:
            response = completion(**completion_params)
            content = response.choices[0].message.content
            cost = self.calculate_cost(response)

            if schema:
                json_output = trim_and_load_json(content)
                return (
                    schema(**json_output),
                    cost,
                )  # Return both the schema instance and cost as defined as native model
            else:
                return content, cost  # Return tuple with cost
        except Exception as e:
            logging.error(f"Error in LiteLLM generation: {str(e)}")
            raise e

    @retry(
        wait=wait_exponential_jitter(
            initial=INITIAL_WAIT, exp_base=EXP_BASE, jitter=JITTER, max=MAX_WAIT
        ),
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:

        from litellm import acompletion

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]

        completion_params = {
            "model": self.name,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
        }

        if self.api_key:
            api_key = require_secret_api_key(
                self.api_key,
                provider_label="LiteLLM",
                env_var_name="LITELLM_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY",
                param_hint="`api_key` to LiteLLMModel(...)",
            )
            completion_params["api_key"] = api_key
        if self.base_url:
            completion_params["api_base"] = self.base_url

        # Add schema if provided
        if schema:
            completion_params["response_format"] = schema

        # Add any additional parameters
        completion_params.update(self.kwargs)
        completion_params.update(self.generation_kwargs)

        try:
            response = await acompletion(**completion_params)
            content = response.choices[0].message.content
            cost = self.calculate_cost(response)

            if schema:
                json_output = trim_and_load_json(content)
                return (
                    schema(**json_output),
                    cost,
                )  # Return both the schema instance and cost as defined as native model
            else:
                return content, cost  # Return tuple with cost
        except Exception as e:
            logging.error(f"Error in LiteLLM async generation: {str(e)}")
            raise e

    @retry(
        wait=wait_exponential_jitter(
            initial=INITIAL_WAIT, exp_base=EXP_BASE, jitter=JITTER, max=MAX_WAIT
        ),
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[Any, float]:
        from litellm import completion

        try:
            api_key = require_secret_api_key(
                self.api_key,
                provider_label="LiteLLM",
                env_var_name="LITELLM_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY",
                param_hint="`api_key` to LiteLLMModel(...)",
            )
            if check_if_multimodal(prompt):
                prompt = convert_to_multi_modal_array(input=prompt)
                content = self.generate_content(prompt)
            else:
                content = [{"type": "text", "text": prompt}]
            completion_params = {
                "model": self.name,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "api_key": api_key,
                "api_base": self.base_url,
                "logprobs": True,
                "top_logprobs": top_logprobs,
            }
            completion_params.update(self.kwargs)
            completion_params.update(self.generation_kwargs)

            response = completion(**completion_params)
            cost = self.calculate_cost(response)
            return response, float(cost)  # Ensure cost is always a float

        except Exception as e:
            logging.error(f"Error in LiteLLM generate_raw_response: {e}")
            return None, 0.0  # Return 0.0 cost on error

    @retry(
        wait=wait_exponential_jitter(
            initial=INITIAL_WAIT, exp_base=EXP_BASE, jitter=JITTER, max=MAX_WAIT
        ),
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    async def a_generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[Any, float]:
        from litellm import acompletion

        try:
            api_key = require_secret_api_key(
                self.api_key,
                provider_label="LiteLLM",
                env_var_name="LITELLM_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY",
                param_hint="`api_key` to LiteLLMModel(...)",
            )
            if check_if_multimodal(prompt):
                prompt = convert_to_multi_modal_array(input=prompt)
                content = self.generate_content(prompt)
            else:
                content = [{"type": "text", "text": prompt}]
            completion_params = {
                "model": self.name,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "api_key": api_key,
                "api_base": self.base_url,
                "logprobs": True,
                "top_logprobs": top_logprobs,
            }
            completion_params.update(self.kwargs)
            completion_params.update(self.generation_kwargs)

            response = await acompletion(**completion_params)
            cost = self.calculate_cost(response)
            return response, float(cost)  # Ensure cost is always a float

        except Exception as e:
            logging.error(f"Error in LiteLLM a_generate_raw_response: {e}")
            return None, 0.0  # Return 0.0 cost on error

    @retry(
        wait=wait_exponential_jitter(
            initial=INITIAL_WAIT, exp_base=EXP_BASE, jitter=JITTER, max=MAX_WAIT
        ),
        stop=stop_after_attempt(MAX_RETRIES),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[List[str], float]:
        from litellm import completion

        try:
            api_key = require_secret_api_key(
                self.api_key,
                provider_label="LiteLLM",
                env_var_name="LITELLM_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|GOOGLE_API_KEY",
                param_hint="`api_key` to LiteLLMModel(...)",
            )
            completion_params = {
                "model": self.name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "n": n,
                "api_key": api_key,
                "api_base": self.base_url,
            }
            completion_params.update(self.kwargs)

            response = completion(**completion_params)
            samples = [choice.message.content for choice in response.choices]
            cost = self.calculate_cost(response)
            return samples, cost

        except Exception as e:
            logging.error(f"Error in LiteLLM generate_samples: {e}")
            raise

    def generate_content(
        self, multimodal_input: List[Union[str, MLLMImage]] = []
    ):
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

    def calculate_cost(self, response: Any) -> float:
        """Calculate the cost of the response based on token usage."""
        try:
            # Get token usage from response
            input_tokens = getattr(response.usage, "prompt_tokens", 0)
            output_tokens = getattr(response.usage, "completion_tokens", 0)

            # Try to get cost from response if available
            if hasattr(response, "cost") and response.cost is not None:
                cost = float(response.cost)
            else:
                # Fallback to token-based calculation
                # Default cost per token (can be adjusted based on provider)
                input_cost_per_token = 0.0001
                output_cost_per_token = 0.0002
                cost = (input_tokens * input_cost_per_token) + (
                    output_tokens * output_cost_per_token
                )

            # Update total evaluation cost
            self.evaluation_cost += float(cost)
            return float(cost)
        except Exception as e:
            logging.warning(f"Error calculating cost: {e}")
            return 0.0

    def get_evaluation_cost(self) -> float:
        """Get the total evaluation cost."""
        return float(self.evaluation_cost)

    def get_model_name(self) -> str:
        from litellm import get_llm_provider

        provider = get_llm_provider(self.name)
        return f"{self.name} ({provider})"

    def load_model(self, async_mode: bool = False):
        """
        LiteLLM doesn't require explicit model loading as it handles client creation
        internally during completion calls. This method is kept for compatibility
        with the DeepEval interface.

        Args:
            async_mode: Whether to use async mode (not used in LiteLLM)

        Returns:
            None as LiteLLM handles client creation internally
        """
        return None

    def supports_multimodal(self):
        return True
