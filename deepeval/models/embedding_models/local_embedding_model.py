from langchain_openai import OpenAIEmbeddings
from typing import List

from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models import DeepEvalBaseEmbeddingModel

class LocalEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.base_url = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_EMBEDDING_BASE_URL)
        self.api_key = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_EMBEDDING_API_KEY)
        model_name = KEY_FILE_HANDLER.fetch_data(KeyValues.LOCAL_EMBEDDING_MODEL_NAME)
        self.args = args
        self.kwargs = kwargs
        super().__init__(model_name)

    def load_model(self):
        return OpenAIEmbeddings(
            model=self.model_name,
            openai_api_base=self.base_url,
            openai_api_key=self.api_key,
            **self.kwargs,
        )

    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return embedding_model.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return embedding_model.embed_documents(texts)

    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_query(text)

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        return await embedding_model.aembed_documents(texts)

    def get_model_name(self):
        return self.model_name
