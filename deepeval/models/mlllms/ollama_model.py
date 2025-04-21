from typing import Optional, Tuple, List, Union, Dict
from ollama import Client, AsyncClient, ChatResponse
from pydantic import BaseModel
import requests
import base64
import io

from deepeval.key_handler import KEY_FILE_HANDLER, KeyValues
from deepeval.models import DeepEvalBaseMLLM
from deepeval.test_case import MLLMImage


class MultimodalOllamaModel(DeepEvalBaseMLLM):
    def __init__(
        self,
    ):
        model_name = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_NAME)
        self.base_url = KEY_FILE_HANDLER.fetch_data(
            KeyValues.LOCAL_MODEL_BASE_URL
        )
        super().__init__(model_name)

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
                messages.append(
                    {
                        "role": "user",
                        "images": [self.convert_to_base64(ele.url, ele.local)],
                    }
                )
        return messages

    ###############################################
    # Utilities
    ###############################################

    def convert_to_base64(self, image_source: str, is_local: bool) -> str:
        from PIL import Image

        try:
            if not is_local:
                response = requests.get(image_source, stream=True)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                image = Image.open(io.BytesIO(response.content))
            else:
                image = Image.open(image_source)

            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str

        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return Client(host=self.base_url)
        else:
            return AsyncClient(host=self.base_url)

    def get_model_name(self):
        return f"{self.model_name} (Ollama)"
