from ollama import Client, AsyncClient
from typing import List

from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models import DeepEvalBaseEmbeddingModel


class OllamaEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self, *args, **kwargs):
        self.base_url = KEY_FILE_HANDLER.fetch_data(
            KeyValues.LOCAL_EMBEDDING_BASE_URL
        )
        model_name = KEY_FILE_HANDLER.fetch_data(
            KeyValues.LOCAL_EMBEDDING_MODEL_NAME
        )
        self.api_key = KEY_FILE_HANDLER.fetch_data(
            KeyValues.LOCAL_EMBEDDING_API_KEY
        )
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return Client(host=self.base_url)

        return AsyncClient(host=self.base_url)

    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        response = embedding_model.embed(
            model=self.model_name,
            input=text,
        )
        return response["embeddings"][0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        response = embedding_model.embed(
            model=self.model_name,
            input=texts,
        )
        return response["embeddings"]

    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model(async_mode=True)
        response = await embedding_model.embed(
            model=self.model_name,
            input=text,
        )
        return response["embeddings"][0]

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model(async_mode=True)
        response = await embedding_model.embed(
            model=self.model_name,
            input=texts,
        )
        return response["embeddings"]

    def get_model_name(self):
        return self.model_name
