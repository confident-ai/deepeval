from pydantic import BaseModel
from google.genai import types
from typing import Optional, Dict
from google import genai

from deepeval.models.retry_policy import (
    create_retry_decorator,
)
from deepeval.key_handler import ModelKeyValues, KEY_FILE_HANDLER
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.constants import ProviderSlug as PS


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
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        model_name = (
            model_name
            or KEY_FILE_HANDLER.fetch_data(ModelKeyValues.GEMINI_MODEL_NAME)
            or default_gemini_model
        )

        # Get API key from key handler if not provided
        self.api_key = api_key or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.GOOGLE_API_KEY
        )
        self.project = project or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.GOOGLE_CLOUD_PROJECT
        )
        self.location = location or KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.GOOGLE_CLOUD_LOCATION
        )
        self.use_vertexai = KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.GOOGLE_GENAI_USE_VERTEXAI
        )
        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_name, **kwargs)

    def should_use_vertexai(self):
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

    def load_model(self, *args, **kwargs):
        """Creates a client.
        With Gen AI SDK, model is set at inference time, so there is no
        model to load and initialize.
        This method name is kept for compatibility with other LLMs.

        Returns:
            A GenerativeModel instance configured for evaluation.
        """
        if self.should_use_vertexai():
            if not self.project or not self.location:
                raise ValueError(
                    "When using Vertex AI API, both project and location are required."
                    "Either provide them as arguments or set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables, "
                    "or set them in your DeepEval configuration."
                )

            # Create client for Vertex AI
            self.client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
                **self.kwargs,
            )
        else:
            if not self.api_key:
                raise ValueError(
                    "Google API key is required. Either provide it directly, set GOOGLE_API_KEY environment variable, "
                    "or set it in your DeepEval configuration."
                )
            # Create client for Gemini API
            self.client = genai.Client(api_key=self.api_key, **self.kwargs)

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
        return self.client.models

    @retry_gemini
    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str:
        """Generates text from a prompt.

        Args:
            prompt: Text prompt
            schema: Optional Pydantic model for structured output

        Returns:
            Generated text response or structured output as Pydantic model
        """
        if schema is not None:
            response = self.client.models.generate_content(
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
            response = self.client.models.generate_content(
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
        if schema is not None:
            response = await self.client.aio.models.generate_content(
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
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=self.model_safety_settings,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                ),
            )
            return response.text, 0

    def get_model_name(self) -> str:
        """Returns the name of the Gemini model being used."""
        return self.model_name
