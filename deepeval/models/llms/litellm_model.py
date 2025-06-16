from typing import Optional, Tuple, Union, Dict, List, Any
from pydantic import BaseModel
import logging
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential_jitter,
    RetryCallState,
)
import os

from deepeval.models import DeepEvalBaseLLM
from deepeval.models.utils import parse_model_name
from deepeval.models.llms.utils import trim_and_load_json
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER


def log_retry_error(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    logging.error(
        f"LiteLLM Error: {exception} Retrying: {retry_state.attempt_number} time(s)..."
    )


# Define retryable exceptions
retryable_exceptions = (
    Exception,  # LiteLLM handles specific exceptions internally
)


class LiteLLMModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0,
        **kwargs,
    ):
        from litellm import completion, acompletion, get_llm_provider

        # Get model name from parameter or key file
        model_name = model or KEY_FILE_HANDLER.fetch_data(
            KeyValues.LITELLM_MODEL_NAME
        )
        if not model_name:
            raise ValueError(
                "Model name must be provided either through parameter or set-litellm command"
            )

        # Get API key from parameter, key file, or environment variable
        self.api_key = (
            api_key
            or KEY_FILE_HANDLER.fetch_data(KeyValues.LITELLM_API_KEY)
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
        )

        # Get API base from parameter, key file, or environment variable
        self.api_base = (
            api_base
            or KEY_FILE_HANDLER.fetch_data(KeyValues.LITELLM_API_BASE)
            or os.getenv("LITELLM_API_BASE")
        )

        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature
        self.kwargs = kwargs
        self.evaluation_cost = 0.0  # Initialize cost to 0.0
        super().__init__(model_name)

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Union[str, Dict, Tuple[str, float]]:
        completion_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        if self.api_key:
            completion_params["api_key"] = self.api_key
        if self.api_base:
            completion_params["api_base"] = self.api_base

        # Add schema if provided
        if schema:
            completion_params["response_format"] = {"type": "json_object"}
            completion_params["json_schema"] = schema.model_json_schema()

        # Add any additional parameters
        completion_params.update(self.kwargs)

        try:
            response = completion(**completion_params)
            content = response.choices[0].message.content
            cost = self.calculate_cost(response)

            if schema:
                json_output = trim_and_load_json(content)
                return schema(**json_output)  # Return just the schema instance
            else:
                return content, cost  # Return tuple with cost
        except Exception as e:
            logging.error(f"Error in LiteLLM generation: {str(e)}")
            raise e

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Union[str, Dict, Tuple[str, float]]:
        completion_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        if self.api_key:
            completion_params["api_key"] = self.api_key
        if self.api_base:
            completion_params["api_base"] = self.api_base

        # Add schema if provided
        if schema:
            completion_params["response_format"] = {"type": "json_object"}
            completion_params["json_schema"] = schema.model_json_schema()

        # Add any additional parameters
        completion_params.update(self.kwargs)

        try:
            response = await acompletion(**completion_params)
            content = response.choices[0].message.content
            cost = self.calculate_cost(response)

            if schema:
                json_output = trim_and_load_json(content)
                return schema(**json_output)  # Return just the schema instance
            else:
                return content, cost  # Return tuple with cost
        except Exception as e:
            logging.error(f"Error in LiteLLM async generation: {str(e)}")
            raise e

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[Any, float]:
        try:
            completion_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "api_key": self.api_key,
                "api_base": self.api_base,
                "logprobs": True,
                "top_logprobs": top_logprobs,
            }
            completion_params.update(self.kwargs)

            response = completion(**completion_params)
            cost = self.calculate_cost(response)
            return response, float(cost)  # Ensure cost is always a float

        except Exception as e:
            logging.error(f"Error in LiteLLM generate_raw_response: {e}")
            return None, 0.0  # Return 0.0 cost on error

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    async def a_generate_raw_response(
        self,
        prompt: str,
        top_logprobs: int = 5,
    ) -> Tuple[Any, float]:
        try:
            completion_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "api_key": self.api_key,
                "api_base": self.api_base,
                "logprobs": True,
                "top_logprobs": top_logprobs,
            }
            completion_params.update(self.kwargs)

            response = await acompletion(**completion_params)
            cost = self.calculate_cost(response)
            return response, float(cost)  # Ensure cost is always a float

        except Exception as e:
            logging.error(f"Error in LiteLLM a_generate_raw_response: {e}")
            return None, 0.0  # Return 0.0 cost on error

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ) -> Tuple[List[str], float]:
        try:
            completion_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "n": n,
                "api_key": self.api_key,
                "api_base": self.api_base,
            }
            completion_params.update(self.kwargs)

            response = completion(**completion_params)
            samples = [choice.message.content for choice in response.choices]
            cost = self.calculate_cost(response)
            return samples, cost

        except Exception as e:
            logging.error(f"Error in LiteLLM generate_samples: {e}")
            raise

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
        provider = get_llm_provider(self.model_name)
        return f"{self.model_name} ({provider})"

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
