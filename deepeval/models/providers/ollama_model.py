from ollama import Client, AsyncClient, ChatResponse
from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER


class OllamaModel(DeepEvalBaseLLM):
    def __init__(
        self,
    ):
        model_name = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_NAME)
        self.base_url = KEY_FILE_HANDLER.fetch_data(
            KeyValues.LOCAL_EMBEDDING_BASE_URL
        )
        super().__init__(model_name)

    ###############################################
    # Other generate functions
    ###############################################

    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        chat_model = self.load_model()
        response: ChatResponse = chat_model.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
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
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        chat_model = self.load_model(async_mode=True)
        model_name = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_MODEL_NAME)
        response: ChatResponse = await chat_model.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
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

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return Client(host=self.base_url)
        else:
            return AsyncClient(host=self.base_url)

    def get_model_name(self):
        return f"{self.model_name} (Ollama)"
