from typing import Optional, List, Union
import requests
from pydantic import BaseModel
from google.genai import types
from google import genai

from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models.base_model import DeepEvalBaseMLLM
from deepeval.test_case import MLLMImage


default_multimodal_gemini_model = "gemini-1.5-pro"


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
        model_name = (
            model_name
            or KEY_FILE_HANDLER.fetch_data(KeyValues.GEMINI_MODEL_NAME)
            or default_multimodal_gemini_model
        )

        # Get API key from key handler if not provided
        self.api_key = api_key or KEY_FILE_HANDLER.fetch_data(
            KeyValues.GOOGLE_API_KEY
        )
        self.project = project or KEY_FILE_HANDLER.fetch_data(
            KeyValues.GOOGLE_CLOUD_PROJECT
        )
        self.location = location or KEY_FILE_HANDLER.fetch_data(
            KeyValues.GOOGLE_CLOUD_LOCATION
        )
        self.use_vertexai = KEY_FILE_HANDLER.fetch_data(
            KeyValues.GOOGLE_GENAI_USE_VERTEXAI
        )

        super().__init__(model_name, *args, **kwargs)
        self.model = self.load_model(*args, **kwargs)

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
                vertexai=True, project=self.project, location=self.location
            )
        else:
            if not self.api_key:
                raise ValueError(
                    "Google API key is required. Either provide it directly, set GOOGLE_API_KEY environment variable, "
                    "or set it in your DeepEval configuration."
                )

            # Create client for Gemini API
            self.client = genai.Client(api_key=self.api_key)

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
        return self.client.models

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
        for ele in multimodal_input:
            if isinstance(ele, str):
                prompt.append(ele)
            elif isinstance(ele, MLLMImage):
                if ele.local:
                    with open(ele.url, "rb") as f:
                        image_data = f.read()
                else:
                    response = requests.get(ele.url)
                    if response.status_code != 200:
                        raise ValueError(f"Failed to download image: {ele.url}")
                    image_data = response.content

                image_part = types.Part.from_bytes(
                    data=image_data, mime_type="image/jpeg"
                )
                prompt.append(image_part)
            else:
                raise ValueError(f"Invalid input type: {type(ele)}")
        return prompt

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
        prompt = self.generate_prompt(multimodal_input)

        if schema is not None:
            response = self.client.models.generate_content(
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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=self.model_safety_settings,
                    temperature=self.model_temperature,
                ),
            )
            return response.text, 0

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
        prompt = self.generate_prompt(multimodal_input)

        if schema is not None:
            response = await self.client.aio.models.generate_content(
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
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=self.model_safety_settings,
                    temperature=self.model_temperature,
                ),
            )
            return response.text, 0

    def get_model_name(self) -> str:
        """Returns the name of the Gemini model being used."""
        return self.model_name
