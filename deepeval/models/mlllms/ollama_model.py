from typing import Optional, Tuple, List, Union, Dict
from ollama import Client, AsyncClient, ChatResponse
from pydantic import BaseModel
import requests
import base64
import io

from deepeval.models.retry_policy import (
    create_retry_decorator,
)
from deepeval.models import DeepEvalBaseMLLM
from deepeval.test_case import MLLMImage
from deepeval.config.settings import get_settings
from deepeval.constants import ProviderSlug as PS


retry_ollama = create_retry_decorator(PS.OLLAMA)


class MultimodalOllamaModel(DeepEvalBaseMLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        **kwargs,
    ):
        """
        Multimodal Ollama model.

        - `model`: Ollama model name (e.g. "llava").
        - `host`: Ollama base URL (e.g. "http://localhost:11434").
        - extra **kwargs are passed through to the underlying Client.
        """
        settings = get_settings()

        # Resolve host/base URL
        self.base_url = (
            host
            or settings.LOCAL_MODEL_BASE_URL
            and str(settings.LOCAL_MODEL_BASE_URL)
        )

        # Resolve model name
        model_name = model or settings.LOCAL_MODEL_NAME

        # Client kwargs
        self.kwargs = kwargs or {}

        super().__init__(model_name)

    @retry_ollama
    def generate(
        self,
        multimodal_input: List[Union[str, MLLMImage]],
        schema: Optional[BaseModel] = None,
    ) -> Tuple[Union[str, Dict], float]:
        chat_model = self.load_model()
        messages = self.generate_messages(multimodal_input)
        response: ChatResponse = chat_model.chat(
            model=self.model_name,
            messages=messages,
            format=schema.model_json_schema() if schema else None,
        )
        return (
            (
                schema.model_validate_json(response.message.content)
                if schema
                else response.message.content
            ),
            0,
        )

    @retry_ollama
    async def a_generate(
        self,
        multimodal_input: List[Union[str, MLLMImage]],
        schema: Optional[BaseModel] = None,
    ) -> Tuple[str, float]:
        chat_model = self.load_model(async_mode=True)
        messages = self.generate_messages(multimodal_input)
        response: ChatResponse = await chat_model.chat(
            model=self.model_name,
            messages=messages,
            format=schema.model_json_schema() if schema else None,
        )
        return (
            (
                schema.model_validate_json(response.message.content)
                if schema
                else response.message.content
            ),
            0,
        )

    def generate_messages(
        self, multimodal_input: List[Union[str, MLLMImage]] = []
    ):
        messages = []
        for ele in multimodal_input:
            if isinstance(ele, str):
                messages.append(
                    {
                        "role": "user",
                        "content": ele,
                    }
                )
            elif isinstance(ele, MLLMImage):
                img_b64 = self.convert_to_base64(ele.url, ele.local)
                if img_b64 is not None:
                    messages.append(
                        {
                            "role": "user",
                            "images": [img_b64],
                        }
                    )
        return messages

    ###############################################
    # Utilities
    ###############################################

    def convert_to_base64(self, image_source: str, is_local: bool) -> str:
        from PIL import Image

        settings = get_settings()
        try:
            if not is_local:
                response = requests.get(
                    image_source,
                    stream=True,
                    timeout=(
                        settings.MEDIA_IMAGE_CONNECT_TIMEOUT_SECONDS,
                        settings.MEDIA_IMAGE_READ_TIMEOUT_SECONDS,
                    ),
                )
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                image = Image.open(io.BytesIO(response.content))
            else:
                image = Image.open(image_source)

            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str

        except (requests.exceptions.RequestException, OSError) as e:
            # Log, then rethrow so @retry_ollama can retry generate_messages() on network failures
            print(f"Image fetch/encode failed: {e}")
            raise
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(Client)
        return self._build_client(AsyncClient)

    def _client_kwargs(self) -> Dict:
        """
        Return client-init kwargs.
        Ollama's Python client doesn't have built-in retry config like OpenAI,
        so we just pass these through untouched.
        """
        return dict(self.kwargs or {})

    def _build_client(self, cls):
        return cls(host=self.base_url, **self._client_kwargs())

    def get_model_name(self):
        return f"{self.model_name} (Ollama)"
