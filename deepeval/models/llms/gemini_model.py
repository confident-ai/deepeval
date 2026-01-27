import json
import base64
from pydantic import BaseModel, SecretStr
from typing import TYPE_CHECKING, Optional, Dict, List, Union, Tuple

from deepeval.errors import DeepEvalError
from deepeval.test_case import MLLMImage
from deepeval.config.settings import get_settings
from deepeval.models.utils import require_secret_api_key
from deepeval.models.retry_policy import (
    create_retry_decorator,
)
from deepeval.utils import (
    convert_to_multi_modal_array,
    check_if_multimodal,
    require_dependency,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.constants import ProviderSlug as PS
from deepeval.models.llms.constants import GEMINI_MODELS_DATA

if TYPE_CHECKING:
    from google.genai import Client

default_gemini_model = "gemini-2.5-pro"

# consistent retry rules
retry_gemini = create_retry_decorator(PS.GOOGLE)


class GeminiModel(DeepEvalBaseLLM):
    """Class that implements Google Gemini models for text-based evaluation.

    This class provides integration with Google's Gemini models through the Google GenAI SDK,
    supporting text-only inputs for evaluation tasks.
    To use Gemini API, set api_key attribute only.
    To use Vertex AI API, set project and location attributes.

    Attributes:
        model: Name of the Gemini model to use
        api_key: Google API key for authentication
        project: Google Cloud project ID
        location: Google Cloud location

    Example:
        ```python
        from deepeval.models import GeminiModel

        # Initialize the model
        model = GeminiModel(
            model="gemini-1.5-pro-001",
            api_key="your-api-key"
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
        project: Optional[str] = None,
        location: Optional[str] = None,
        service_account_key: Optional[Union[str, Dict[str, str]]] = None,
        use_vertexai: Optional[bool] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):

        settings = get_settings()

        model = model or settings.GEMINI_MODEL_NAME or default_gemini_model
        self.model_data = GEMINI_MODELS_DATA.get(model)

        # Get API key from settings if not provided
        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and aolike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.GOOGLE_API_KEY

        if temperature is not None:
            temperature = float(temperature)
        elif settings.TEMPERATURE is not None:
            temperature = settings.TEMPERATURE
        else:
            temperature = 0.0

        self.project = project or settings.GOOGLE_CLOUD_PROJECT
        location = (
            location if location is not None else settings.GOOGLE_CLOUD_LOCATION
        )
        self.location = str(location).strip() if location is not None else None
        self.use_vertexai = (
            use_vertexai
            if use_vertexai is not None
            else settings.GOOGLE_GENAI_USE_VERTEXAI
        )

        self.service_account_key: Optional[SecretStr] = None
        if service_account_key is None:
            self.service_account_key = settings.GOOGLE_SERVICE_ACCOUNT_KEY
        elif isinstance(service_account_key, dict):
            self.service_account_key = SecretStr(
                json.dumps(service_account_key)
            )
        else:
            str_value = str(service_account_key).strip()
            self.service_account_key = (
                SecretStr(str_value) if str_value else None
            )

        if temperature < 0:
            raise DeepEvalError("Temperature must be >= 0.")

        self.temperature = temperature

        # Raw kwargs destined for the underlying Client
        self.kwargs = kwargs
        self.kwargs.pop("temperature", None)

        self.generation_kwargs = dict(generation_kwargs or {})
        self.generation_kwargs.pop("temperature", None)

        self._module = self._require_module()
        # Configure default model generation settings
        self.model_safety_settings = [
            self._module.types.SafetySetting(
                category=self._module.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=self._module.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            self._module.types.SafetySetting(
                category=self._module.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=self._module.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            self._module.types.SafetySetting(
                category=self._module.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=self._module.types.HarmBlockThreshold.BLOCK_NONE,
            ),
            self._module.types.SafetySetting(
                category=self._module.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=self._module.types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

        super().__init__(model)

    def should_use_vertexai(self) -> bool:
        """Checks if the model should use Vertex AI for generation.

        This is determined first by the value of `GOOGLE_GENAI_USE_VERTEXAI`
        environment variable. If not set, it checks for the presence of the
        project and location.

        Returns:
            True if the model should use Vertex AI, False otherwise
        """
        if self.use_vertexai is not None:
            return self.use_vertexai
        if self.project and self.location:
            return True
        else:
            return False

    @retry_gemini
    def generate_content(
        self, multimodal_input: Optional[List[Union[str, MLLMImage]]] = None
    ):
        multimodal_input = (
            multimodal_input if multimodal_input is not None else []
        )
        content = []

        for element in multimodal_input:
            if isinstance(element, str):
                content.append(element)
            elif isinstance(element, MLLMImage):
                # Gemini doesn't support direct external URLs
                # Must convert all images to bytes
                if element.url and not element.local:
                    import requests

                    settings = get_settings()

                    response = requests.get(
                        element.url,
                        timeout=(
                            settings.MEDIA_IMAGE_CONNECT_TIMEOUT_SECONDS,
                            settings.MEDIA_IMAGE_READ_TIMEOUT_SECONDS,
                        ),
                    )
                    response.raise_for_status()
                    image_data = response.content
                    mime_type = response.headers.get(
                        "content-type", element.mimeType or "image/jpeg"
                    )
                else:
                    element.ensure_images_loaded()
                    try:
                        image_data = base64.b64decode(element.dataBase64)
                    except Exception:
                        raise ValueError(
                            f"Invalid base64 data in MLLMImage: {element._id}"
                        )

                    mime_type = element.mimeType or "image/jpeg"

                # Create Part from bytes
                image_part = self._module.types.Part.from_bytes(
                    data=image_data, mime_type=mime_type
                )
                content.append(image_part)
            else:
                raise DeepEvalError(f"Invalid input type: {type(element)}")

        return content

    ###############################################
    # Generate functions
    ###############################################

    @retry_gemini
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        """Generates text from a prompt.

        Args:
            prompt: Text prompt
            schema: Optional Pydantic model for structured output

        Returns:
            Generated text response or structured output as Pydantic model
        """
        client = self.load_model()

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(prompt)
            prompt = self.generate_content(prompt)

        if schema is not None:
            response = client.models.generate_content(
                model=self.name,
                contents=prompt,
                config=self._module.types.GenerateContentConfig(
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
                model=self.name,
                contents=prompt,
                config=self._module.types.GenerateContentConfig(
                    safety_settings=self.model_safety_settings,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                ),
            )
            return response.text, 0

    @retry_gemini
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, BaseModel], float]:
        """Asynchronously generates text from a prompt.

        Args:
            prompt: Text prompt
            schema: Optional Pydantic model for structured output

        Returns:
            Generated text response or structured output as Pydantic model
        """
        client = self.load_model()

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(prompt)
            prompt = self.generate_content(prompt)

        if schema is not None:
            response = await client.aio.models.generate_content(
                model=self.name,
                contents=prompt,
                config=self._module.types.GenerateContentConfig(
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
                model=self.name,
                contents=prompt,
                config=self._module.types.GenerateContentConfig(
                    safety_settings=self.model_safety_settings,
                    temperature=self.temperature,
                    **self.generation_kwargs,
                ),
            )
            return response.text, 0

    #########################
    # Capabilities          #
    #########################

    def supports_log_probs(self) -> Union[bool, None]:
        return self.model_data.supports_log_probs

    def supports_temperature(self) -> Union[bool, None]:
        return self.model_data.supports_temperature

    def supports_multimodal(self) -> Union[bool, None]:
        return self.model_data.supports_multimodal

    def supports_structured_outputs(self) -> Union[bool, None]:
        """
        OpenAI models that natively enforce typed structured outputs.
         Used by generate(...) when a schema is provided.
        """
        return self.model_data.supports_structured_outputs

    def supports_json_mode(self) -> Union[bool, None]:
        """
        OpenAI models that enforce JSON mode
        """
        return self.model_data.supports_json

    #########
    # Model #
    #########

    def load_model(self):
        """Creates a client.
        With Gen AI SDK, model is set at inference time, so there is no
        model to load and initialize.
        This method name is kept for compatibility with other LLMs.

        Returns:
            A GenerativeModel instance configured for evaluation.
        """
        return self._build_client()

    def _require_oauth2(self):
        return require_dependency(
            "google.oauth2",
            provider_label="GeminiModel",
            install_hint="Install it with `pip install google-auth`.",
        )

    def _require_module(self):
        return require_dependency(
            "google.genai",
            provider_label="GeminiModel",
            install_hint="Install it with `pip install google-genai`.",
        )

    def _client_kwargs(self, **override_kwargs) -> Dict:
        """Merge ctor kwargs with any overrides passed at load_model time."""
        client_kwargs = dict(self.kwargs or {})
        if override_kwargs:
            client_kwargs.update(override_kwargs)
        return client_kwargs

    def _build_client(self) -> "Client":
        client_kwargs = self._client_kwargs(**self.kwargs)

        if self.should_use_vertexai():
            if not self.project or not self.location:
                raise DeepEvalError(
                    "When using Vertex AI API, both project and location are required. "
                    "Either provide them as arguments or set GOOGLE_CLOUD_PROJECT and "
                    "GOOGLE_CLOUD_LOCATION in your DeepEval configuration."
                )

            # if no service account key is provided, allow the SDK
            # to resolve Application Default Credentials automatically.
            credentials = None
            if self.service_account_key is not None:
                service_account_key_json = require_secret_api_key(
                    self.service_account_key,
                    provider_label="Google Gemini",
                    env_var_name="GOOGLE_SERVICE_ACCOUNT_KEY",
                    param_hint="`service_account_key` to GeminiModel(...)",
                )

                try:
                    service_account_key = json.loads(service_account_key_json)
                except Exception as e:
                    raise DeepEvalError(
                        "GOOGLE_SERVICE_ACCOUNT_KEY must be valid JSON for a Google service account."
                    ) from e

                if not isinstance(service_account_key, dict):
                    raise DeepEvalError(
                        "GOOGLE_SERVICE_ACCOUNT_KEY must decode to a JSON object."
                    )

                oauth2 = self._require_oauth2()
                credentials = oauth2.service_account.Credentials.from_service_account_info(
                    service_account_key,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )

            client = self._module.Client(
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

            client = self._module.Client(api_key=api_key, **client_kwargs)

        return client

    def get_model_name(self):
        return f"{self.name} (Gemini)"
