import json

from pydantic import BaseModel, SecretStr
from google.genai import types, Client
from typing import Optional, Dict

from deepeval.config.settings import get_settings
from deepeval.models.utils import require_secret_api_key
from deepeval.models.retry_policy import (
    create_retry_decorator,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.constants import ProviderSlug as PS
from google.oauth2 import service_account

default_gemini_model = "gemini-1.5-pro"

# consistent retry rules
retry_gemini = create_retry_decorator(PS.GOOGLE)


class GeminiModel(DeepEvalBaseLLM):
    """Class that implements Google Gemini models for text-based evaluation.

    This class provides integration with Google's Gemini models through the Google GenAI SDK,
    supporting text-only inputs for evaluation tasks.
    To use Gemini API, set api_key attribute only.
    To use Vertex AI API, set project and location attributes.

    Attributes:
        model_name: Name of the Gemini model to use
        api_key: Google API key for authentication
        project: Google Cloud project ID
        location: Google Cloud location

    Example:
        ```python
        from deepeval.models import GeminiModel

        # Initialize the model
        model = GeminiModel(
            model_name="gemini-1.5-pro-001",
            api_key="your-api-key"
        )

        # Generate text
        response = model.generate("What is the capital of France?")
        ```
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        service_account_key: Optional[Dict[str, str]] = None,
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):

        settings = get_settings()

        model_name = (
            model_name or settings.GEMINI_MODEL_NAME or default_gemini_model
        )

        # Get API key from settings if not provided
        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and aolike
            self.api_key: SecretStr | None = SecretStr(api_key)
        else:
            self.api_key = settings.GOOGLE_API_KEY

        self.project = project or settings.GOOGLE_CLOUD_PROJECT
        self.location = (
            location
            or settings.GOOGLE_CLOUD_LOCATION is not None
            and str(settings.GOOGLE_CLOUD_LOCATION)
        )
        self.use_vertexai = settings.GOOGLE_GENAI_USE_VERTEXAI

        if service_account_key:
            self.service_account_key = service_account_key
        else:
            service_account_key_data = settings.GOOGLE_SERVICE_ACCOUNT_KEY
            if service_account_key_data is None:
                self.service_account_key = None
            elif isinstance(service_account_key_data, str):
                self.service_account_key = json.loads(service_account_key_data)

        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature

        # Raw kwargs destined for the underlying Client
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}

        # Configure default model generation settings
        self.model_safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        super().__init__(model_name, **kwargs)

    def should_use_vertexai(self) -> bool:
        """Checks if the model should use Vertex AI for generation.

        This is determined first by the value of `GOOGLE_GENAI_USE_VERTEXAI`
        environment variable. If not set, it checks for the presence of the
        project and location.

        Returns:
            True if the model should use Vertex AI, False otherwise
        """
        if self.use_vertexai is not None:
            return self.use_vertexai.lower() == "yes"
        if self.project and self.location:
            return True
        else:
            return False

    ###############################################
    # Generate functions
    ###############################################

    @retry_gemini
    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str:
        """Generates text from a prompt.

        Args:
            prompt: Text prompt
            schema: Optional Pydantic model for structured output

        Returns:
            Generated text response or structured output as Pydantic model
        """
        client = self.load_model()

        if schema is not None:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    safety_settings=self.model_safety_settings,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                ),
            )
            return response.parsed, 0
        else:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=self.model_safety_settings,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                ),
            )
            return response.text, 0

    @retry_gemini
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> str:
        """Asynchronously generates text from a prompt.

        Args:
            prompt: Text prompt
            schema: Optional Pydantic model for structured output

        Returns:
            Generated text response or structured output as Pydantic model
        """
        client = self.load_model()

        if schema is not None:
            response = await client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    safety_settings=self.model_safety_settings,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                ),
            )
            return response.parsed, 0
        else:
            response = await client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=self.model_safety_settings,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                ),
            )
            return response.text, 0

    #########
    # Model #
    #########

    def get_model_name(self) -> str:
        """Returns the name of the Gemini model being used."""
        return self.model_name

    def load_model(self, *args, **kwargs):
        """Creates a client.
        With Gen AI SDK, model is set at inference time, so there is no
        model to load and initialize.
        This method name is kept for compatibility with other LLMs.

        Returns:
            A GenerativeModel instance configured for evaluation.
        """
        return self._build_client(**kwargs)

    def _client_kwargs(self, **override_kwargs) -> Dict:
        """Merge ctor kwargs with any overrides passed at load_model time."""
        client_kwargs = dict(self.kwargs or {})
        if override_kwargs:
            client_kwargs.update(override_kwargs)
        return client_kwargs

    def _build_client(self, **override_kwargs) -> Client:
        client_kwargs = self._client_kwargs(**override_kwargs)

        if self.should_use_vertexai():
            if not self.project or not self.location:
                raise ValueError(
                    "When using Vertex AI API, both project and location are required. "
                    "Either provide them as arguments or set GOOGLE_CLOUD_PROJECT and "
                    "GOOGLE_CLOUD_LOCATION in your DeepEval configuration."
                )

            credentials = (
                service_account.Credentials.from_service_account_info(
                    self.service_account_key,
                    scopes=[
                        "https://www.googleapis.com/auth/cloud-platform",
                    ],
                )
                if self.service_account_key
                else None
            )

            client = Client(
                vertexai=True,
                project=self.project,
                location=self.location,
                credentials=credentials,
                **client_kwargs,
            )
        else:
            api_key = require_secret_api_key(
                self.api_key,
                provider_label="Google Gemini",
                env_var_name="GOOGLE_API_KEY",
                param_hint="`api_key` to GeminiModel(...)",
            )

            client = Client(api_key=api_key, **client_kwargs)

        return client
