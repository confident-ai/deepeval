import requests
from typing import Optional, List, Union
from pydantic import BaseModel, SecretStr
from google.genai import types
from google import genai

from deepeval.config.settings import get_settings
from deepeval.models.utils import require_secret_api_key
from deepeval.models.retry_policy import (
    create_retry_decorator,
)
from deepeval.models.base_model import DeepEvalBaseMLLM
from deepeval.test_case import MLLMImage
from deepeval.constants import ProviderSlug as PS


default_multimodal_gemini_model = "gemini-1.5-pro"
# consistent retry rules
retry_gemini = create_retry_decorator(PS.GOOGLE)


class MultimodalGeminiModel(DeepEvalBaseMLLM):
    """Class that implements Google Gemini models for multimodal evaluation.

    This class provides integration with Google's Gemini models through the Google GenAI SDK,
    supporting both text and multimodal (text + image) inputs for evaluation tasks.
    To use Gemini API, set api_key attribute only.
    To use Vertex AI API, set project and location attributes.

    Attributes:
        model_name: Name of the Gemini model to use
        api_key: Google API key for authentication
        project: Google Cloud project ID
        location: Google Cloud location

    Example:
        ```python
        from deepeval.models import MultimodalGeminiModel

        # Initialize the model
        model = MultimodalGeminiModel(
            model_name="gemini-pro-vision",
            api_key="your-api-key"
        )

        # Generate text from text + image input
        response = model.generate([
            "Describe what you see in this image:",
            MLLMImage(url="path/to/image.jpg", local=True)
        ])
        ```
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        *args,
        **kwargs,
    ):
        settings = get_settings()
        model_name = (
            model_name
            or settings.GEMINI_MODEL_NAME
            or default_multimodal_gemini_model
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

        # Keep any extra kwargs for the underlying genai.Client
        self.args = args
        self.kwargs = kwargs

        # Configure default model generation settings
        self.model_safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            ),
        ]
        self.model_temperature = 0.0

        super().__init__(model_name, *args, **kwargs)

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

    # TODO: Refactor generate prompt to minimize the work done on retry
    @retry_gemini
    def generate_prompt(
        self, multimodal_input: List[Union[str, MLLMImage]] = []
    ) -> List[Union[str, MLLMImage]]:
        """Converts DeepEval multimodal input into GenAI SDK compatible format.

        Args:
            multimodal_input: List of strings and MLLMImage objects

        Returns:
            List of strings and PIL Image objects ready for model input

        Raises:
            ValueError: If an invalid input type is provided
        """
        prompt = []
        settings = get_settings()

        for ele in multimodal_input:
            if isinstance(ele, str):
                prompt.append(ele)
            elif isinstance(ele, MLLMImage):
                if ele.local:
                    with open(ele.url, "rb") as f:
                        image_data = f.read()
                else:
                    response = requests.get(
                        ele.url,
                        timeout=(
                            settings.MEDIA_IMAGE_CONNECT_TIMEOUT_SECONDS,
                            settings.MEDIA_IMAGE_READ_TIMEOUT_SECONDS,
                        ),
                    )
                    response.raise_for_status()
                    image_data = response.content

                image_part = types.Part.from_bytes(
                    data=image_data, mime_type="image/jpeg"
                )
                prompt.append(image_part)
            else:
                raise ValueError(f"Invalid input type: {type(ele)}")
        return prompt

    @retry_gemini
    def generate(
        self,
        multimodal_input: List[Union[str, MLLMImage]],
        schema: Optional[BaseModel] = None,
    ) -> str:
        """Generates text from multimodal input.

        Args:
            multimodal_input: List of strings and MLLMImage objects
            schema: Optional Pydantic model for structured output

        Returns:
            Generated text response
        """
        client = self.load_model()
        prompt = self.generate_prompt(multimodal_input)

        if schema is not None:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    safety_settings=self.model_safety_settings,
                    temperature=self.model_temperature,
                ),
            )
            return response.parsed, 0
        else:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=self.model_safety_settings,
                    temperature=self.model_temperature,
                ),
            )
            return response.text, 0

    @retry_gemini
    async def a_generate(
        self,
        multimodal_input: List[Union[str, MLLMImage]],
        schema: Optional[BaseModel] = None,
    ) -> str:
        """Asynchronously generates text from multimodal input.

        Args:
            multimodal_input: List of strings and MLLMImage objects
            schema: Optional Pydantic model for structured output

        Returns:
            Generated text response
        """
        client = self.load_model()
        prompt = self.generate_prompt(multimodal_input)

        if schema is not None:
            response = await client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    safety_settings=self.model_safety_settings,
                    temperature=self.model_temperature,
                ),
            )
            return response.parsed, 0
        else:
            response = await client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=self.model_safety_settings,
                    temperature=self.model_temperature,
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
        """Creates and returns a GenAI client.

        With the Gen AI SDK, the model is set at inference time, so we only
        construct the client here. Kept for compatibility with other MLLMs.
        """
        return self._build_client(**kwargs)

    def _client_kwargs(self, **override_kwargs) -> dict:
        """
        Return kwargs forwarded to genai.Client.

        Start from the ctor kwargs captured on `self.kwargs`, then apply any
        overrides passed via load_model(...).
        """
        client_kwargs = dict(self.kwargs or {})
        if override_kwargs:
            client_kwargs.update(override_kwargs)
        return client_kwargs

    def _build_client(self, **override_kwargs):
        """Build and return a genai.Client for either Gemini API or Vertex AI."""
        client_kwargs = self._client_kwargs(**override_kwargs)

        if self.should_use_vertexai():
            if not self.project or not self.location:
                raise ValueError(
                    "When using Vertex AI API, both project and location are required."
                    "Either provide them as arguments or set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables, "
                    "or set them in your DeepEval configuration."
                )

            # Create client for Vertex AI
            return genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
                **client_kwargs,
            )

        api_key = require_secret_api_key(
            self.api_key,
            provider_label="Google Gemini",
            env_var_name="GOOGLE_API_KEY",
            param_hint="`api_key` to MultimodalGeminiModel(...)",
        )

        # Create client for Gemini API
        return genai.Client(api_key=api_key, **client_kwargs)
