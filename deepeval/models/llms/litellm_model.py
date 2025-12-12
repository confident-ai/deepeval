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
import base64
from deepeval.config.settings import get_settings
from deepeval.models.utils import (
    require_secret_api_key,
    normalize_kwargs_and_extract_aliases,
)
from deepeval.test_case import MLLMImage
from deepeval.utils import check_if_multimodal, convert_to_multi_modal_array
from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json


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
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):

        normalized_kwargs, alias_values = normalize_kwargs_and_extract_aliases(
            "LiteLLMModel",
            kwargs,
            _ALIAS_MAP,
        )

        # re-map depricated keywords to re-named positional args
        if base_url is None and "base_url" in alias_values:
            base_url = alias_values["base_url"]

        settings = get_settings()
        # Get model name from parameter or key file
        model = model or settings.LITELLM_MODEL_NAME
        if not model:
            raise ValueError(
                "Model name must be provided either through parameter or set-litellm command"
            )

        # Get API key from parameter, or settings
        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and aolike
            self.api_key: SecretStr | None = SecretStr(api_key)
        else:
            self.api_key = (
                settings.LITELLM_API_KEY
                or settings.LITELLM_PROXY_API_KEY
                or settings.OPENAI_API_KEY
                or settings.ANTHROPIC_API_KEY
                or settings.GOOGLE_API_KEY
            )

        # Get API base from parameter, key file, or environment variable
        self.base_url = (
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

        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature
        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = normalized_kwargs
        self.generation_kwargs = generation_kwargs or {}
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
    ) -> Union[str, Dict, Tuple[str, float]]:

        from litellm import completion

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self._generate_payload(prompt)
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
    ) -> Union[str, Dict, Tuple[str, float]]:

        from litellm import acompletion

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self._generate_payload(prompt)
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
                content = self._generate_payload(prompt)
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
                content = self._generate_payload(prompt)
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

    def _generate_payload(self, multimodal_input):
        """
        Converts multimodal input (text + images) into LiteLLM-compatible content.
        Images are converted to Base64.
        """
        content = []
        for ele in multimodal_input:
            if isinstance(ele, str):
                content.append({"type": "text", "text": ele})
            elif isinstance(ele, MLLMImage):
                mime, raw_bytes = self._parse_image(ele)
                b64 = base64.b64encode(raw_bytes).decode("utf-8")
                content.append({
                    "type": "image", 
                    "image_url": f"data:{mime};base64,{b64}"
                })
        return content
    
    def _parse_image(self, image: MLLMImage):
        if image.dataBase64:
            mime = image.mimeType or "image/jpeg"
            return mime, base64.b64decode(image.dataBase64)

        if image.local and image.filename:
            from PIL import Image
            from io import BytesIO
            img = Image.open(image.filename)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            buf = BytesIO()
            fmt = (image.mimeType or "image/jpeg").split("/")[-1].lower()
            fmt = "jpeg" if fmt == "jpg" else fmt
            img.save(buf, format=fmt.upper())
            return f"image/{fmt}", buf.getvalue()

        if image.url:
            import requests
            resp = requests.get(image.url)
            resp.raise_for_status()
            mime = resp.headers.get("content-type", image.mimeType or "image/jpeg")
            return mime, resp.content

        raise ValueError(
            "MLLMImage must contain dataBase64, or (local=True + filename), or url."
        )

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
