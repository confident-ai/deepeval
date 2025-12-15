from typing import TYPE_CHECKING, Optional, Tuple, Union, Dict, List
from pydantic import BaseModel
import base64

from deepeval.config.settings import get_settings
from deepeval.utils import require_dependency
from deepeval.models.retry_policy import (
    create_retry_decorator,
)
from deepeval.utils import convert_to_multi_modal_array, check_if_multimodal
from deepeval.test_case import MLLMImage
from deepeval.models import DeepEvalBaseLLM
from deepeval.constants import ProviderSlug as PS
from deepeval.models.llms.constants import OLLAMA_MODELS_DATA

if TYPE_CHECKING:
    from ollama import ChatResponse

retry_ollama = create_retry_decorator(PS.OLLAMA)


class OllamaModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()
        model = model or settings.LOCAL_MODEL_NAME
        self.model_data = OLLAMA_MODELS_DATA.get(model)
        self.base_url = (
            base_url
            or (
                settings.LOCAL_MODEL_BASE_URL
                and str(settings.LOCAL_MODEL_BASE_URL)
            )
            or "http://localhost:11434"
        )
        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature
        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model)

    ###############################################
    # Other generate functions
    ###############################################

    @retry_ollama
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        chat_model = self.load_model()

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(prompt)
            messages = self.generate_messages(prompt)
        else:
            messages = [{"role": "user", "content": prompt}]

        response: ChatResponse = chat_model.chat(
            model=self.name,
            messages=messages,
            format=schema.model_json_schema() if schema else None,
            options={
                **{"temperature": self.temperature},
                **self.generation_kwargs,
            },
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
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        chat_model = self.load_model(async_mode=True)

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(prompt)
            messages = self.generate_messages(prompt)
        else:
            messages = [{"role": "user", "content": prompt}]

        response: ChatResponse = await chat_model.chat(
            model=self.name,
            messages=messages,
            format=schema.model_json_schema() if schema else None,
            options={
                **{"temperature": self.temperature},
                **self.generation_kwargs,
            },
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

        for element in multimodal_input:
            if isinstance(element, str):
                messages.append(
                    {
                        "role": "user",
                        "content": element,
                    }
                )
            elif isinstance(element, MLLMImage):
                if element.url and not element.local:
                    import requests
                    from PIL import Image
                    import io

                    settings = get_settings()
                    try:
                        response = requests.get(
                            element.url,
                            stream=True,
                            timeout=(
                                settings.MEDIA_IMAGE_CONNECT_TIMEOUT_SECONDS,
                                settings.MEDIA_IMAGE_READ_TIMEOUT_SECONDS,
                            ),
                        )
                        response.raise_for_status()

                        # Convert to JPEG and encode
                        image = Image.open(io.BytesIO(response.content))
                        buffered = io.BytesIO()

                        # Convert RGBA/LA/P to RGB for JPEG
                        if image.mode in ("RGBA", "LA", "P"):
                            image = image.convert("RGB")

                        image.save(buffered, format="JPEG")
                        img_b64 = base64.b64encode(buffered.getvalue()).decode()

                    except (requests.exceptions.RequestException, OSError) as e:
                        print(f"Image fetch/encode failed: {e}")
                        raise
                else:
                    element.ensure_images_loaded()
                    img_b64 = element.dataBase64

                messages.append(
                    {
                        "role": "user",
                        "images": [img_b64],
                    }
                )

        return messages

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        ollama = require_dependency(
            "ollama",
            provider_label="OllamaModel",
            install_hint="Install it with `pip install ollama`.",
        )
        if not async_mode:
            return self._build_client(ollama.Client)
        return self._build_client(ollama.AsyncClient)

    def _client_kwargs(self) -> Dict:
        """Return kwargs forwarded to the underlying Ollama Client/AsyncClient."""
        return dict(self.kwargs or {})

    def _build_client(self, cls):
        kw = dict(
            host=self.base_url,
            **self._client_kwargs(),
        )
        return cls(**kw)

    def supports_multimodal(self):
        return self.model_data.supports_multimodal

    def get_model_name(self):
        return f"{self.name} (Ollama)"
