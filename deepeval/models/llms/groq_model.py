import os
from pydantic import BaseModel, SecretStr
from typing import TYPE_CHECKING, Optional, Tuple, Union

from deepeval.errors import DeepEvalError
from deepeval.config.settings import get_settings
from deepeval.models.utils import require_secret_api_key
from deepeval.models.retry_policy import create_retry_decorator
from deepeval.utils import require_dependency
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.constants import ProviderSlug
from deepeval.models.llms.constants import GROQ_MODELS_DATA, make_model_data

if TYPE_CHECKING:
    from groq import Groq, AsyncGroq

# -----------------------------------------------------------------------------
# Constants & Defaults
# -----------------------------------------------------------------------------
default_groq_model = "llama3-8b-8192"

# Use a standard string for the retry decorator if ProviderSlug.GROQ doesn't exist yet
retry_groq = create_retry_decorator(ProviderSlug.GROQ)


# -----------------------------------------------------------------------------
# Model Implementation
# -----------------------------------------------------------------------------
class GroqModel(DeepEvalBaseLLM):
    """Class that implements Groq's LPU inference engine for high-speed evaluation.

    This class provides native integration with Groq's ultra-fast API, supporting
    both text generation and structured JSON outputs using Pydantic schemas.

    Attributes:
        model: Name of the Groq model to use (e.g., 'llama3-8b-8192', 'mixtral-8x7b-32768')
        api_key: Groq API key for authentication
        temperature: Sampling temperature for generation

    Example:
        ```python
        from deepeval.models import GroqModel

        # Initialize the model
        model = GroqModel(
            model="llama3-70b-8192",
            api_key="gsk_..."
        )

        # Generate text
        response = model.generate("What is the capital of France?")
        ```
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        settings = get_settings()
        self.model_name = model or default_groq_model

        # Secure API Key handling: Check params -> Settings -> Environment
        if api_key is not None:
            self.api_key = (
                api_key
                if isinstance(api_key, SecretStr)
                else SecretStr(api_key)
            )
        else:
            env_key = getattr(settings, "GROQ_API_KEY", None) or os.environ.get(
                "GROQ_API_KEY"
            )
            self.api_key = (
                env_key
                if isinstance(env_key, SecretStr)
                else (SecretStr(env_key) if env_key else None)
            )

        # Temperature handling
        if temperature is not None:
            self.temperature = float(temperature)
        elif settings.TEMPERATURE is not None:
            self.temperature = settings.TEMPERATURE
        else:
            self.temperature = 0.0

        if self.temperature < 0:
            raise DeepEvalError("Temperature must be >= 0.")

        self.kwargs = kwargs
        self.model_data = GROQ_MODELS_DATA.get(
            self.model_name,
            make_model_data(
                supports_log_probs=False,
                supports_multimodal=False,
                supports_structured_outputs=True,
                supports_json=True,
                input_price=None,
                output_price=None,
            ),
        )
        self._module = self._require_module()

        # Client caching for performance optimization
        self._client: Optional["Groq"] = None
        self._async_client: Optional["AsyncGroq"] = None
        super().__init__(self.model_name)

    def _require_module(self):
        """Lazy loads the groq library to prevent import errors for non-Groq users."""
        return require_dependency(
            "groq",
            provider_label="GroqModel",
            install_hint="Install it with `pip install groq`.",
        )

    def load_model(self, async_mode: bool = False):
        """Initializes and caches the Groq client."""
        if async_mode:
            if self._async_client is not None:
                return self._async_client
        else:
            if self._client is not None:
                return self._client

        api_key = require_secret_api_key(
            self.api_key,
            provider_label="Groq",
            env_var_name="GROQ_API_KEY",
            param_hint="`api_key` to GroqModel(...)",
        )

        if hasattr(api_key, "get_secret_value"):
            api_key = api_key.get_secret_value()

        if async_mode:
            self._async_client = self._module.AsyncGroq(
                api_key=api_key, **self.kwargs
            )
            return self._async_client
        else:
            self._client = self._module.Groq(api_key=api_key, **self.kwargs)
            return self._client

    # -------------------------------------------------------------------------
    # Generation Methods
    # -------------------------------------------------------------------------
    @retry_groq
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        """Generates text or structured output from a prompt."""
        client = self.load_model(async_mode=False)

        chat_args = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        if schema is not None:
            chat_args["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**chat_args)
        content = response.choices[0].message.content

        if schema is not None:
            return schema.model_validate_json(content), 0.0

        return content, 0.0

    @retry_groq
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        """Asynchronously generates text or structured output from a prompt."""
        async_client = self.load_model(async_mode=True)

        chat_args = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        if schema is not None:
            chat_args["response_format"] = {"type": "json_object"}

        response = await async_client.chat.completions.create(**chat_args)
        content = response.choices[0].message.content

        if schema is not None:
            return schema.model_validate_json(content), 0.0

        return content, 0.0

    # -------------------------------------------------------------------------
    # Capabilities
    # -------------------------------------------------------------------------

    def supports_log_probs(self) -> Union[bool, None]:
        return self.model_data.supports_log_probs

    def supports_temperature(self) -> Union[bool, None]:
        # Uses getattr fallback because supports_temperature is not in make_model_data
        return getattr(self.model_data, "supports_temperature", True)

    def supports_multimodal(self) -> Union[bool, None]:
        return self.model_data.supports_multimodal

    def supports_structured_outputs(self) -> Union[bool, None]:
        return self.model_data.supports_structured_outputs

    def supports_json_mode(self) -> Union[bool, None]:
        # Note: The property on make_model_data is called 'supports_json'
        return self.model_data.supports_json

    def get_model_name(self) -> str:
        return f"{self.name} (Groq)"
