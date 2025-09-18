from ollama import Client, AsyncClient
from typing import List

from deepeval.key_handler import EmbeddingKeyValues, KEY_FILE_HANDLER
from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.models.retry_policy import (
    create_retry_decorator,
)
from deepeval.constants import ProviderSlug as PS


retry_ollama = create_retry_decorator(PS.OLLAMA)


class OllamaEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self, *args, **kwargs):
        self.base_url = KEY_FILE_HANDLER.fetch_data(
            EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL
        )
        model_name = KEY_FILE_HANDLER.fetch_data(
            EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME
        )
        # TODO: This is not being used. Clean it up in consistency PR
        self.api_key = KEY_FILE_HANDLER.fetch_data(
            EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY
        )
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    @retry_ollama
    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        response = embedding_model.embed(
            model=self.model_name,
            input=text,
        )
        return response["embeddings"][0]

    @retry_ollama
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        response = embedding_model.embed(
            model=self.model_name,
            input=texts,
        )
        return response["embeddings"]

    @retry_ollama
    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model(async_mode=True)
        response = await embedding_model.embed(
            model=self.model_name,
            input=text,
        )
        return response["embeddings"][0]

    @retry_ollama
    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model(async_mode=True)
        response = await embedding_model.embed(
            model=self.model_name,
            input=texts,
        )
        return response["embeddings"]

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(Client)
        return self._build_client(AsyncClient)

    def _build_client(self, cls):
        return cls(host=self.base_url, **self.kwargs)

    def get_model_name(self):
        return f"{self.model_name} (Ollama)"
