from openai import OpenAI
from typing import List

from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models import DeepEvalBaseEmbeddingModel


class LocalEmbeddingModel(DeepEvalBaseEmbeddingModel):
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

    def load_model(self):
        return OpenAI(base_url=self.base_url, api_key=self.api_key)

    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        response = embedding_model.embeddings.create(
            model=self.model_name,
            input=[text],
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        response = embedding_model.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [data.embedding for data in response.data]

    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        response = embedding_model.embeddings.create(
            model=self.model_name,
            input=[text],
        )
        return response.data[0].embedding

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        response = embedding_model.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [data.embedding for data in response.data]

    def get_model_name(self):
        return self.model_name
