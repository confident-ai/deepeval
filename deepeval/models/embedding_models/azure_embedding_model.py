from langchain_openai import AzureOpenAIEmbeddings
from typing import List

from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models import DeepEvalBaseEmbeddingModel


class AzureOpenAIEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self, *args, **kwargs):
        self.azure_openai_api_key = KEY_FILE_HANDLER.fetch_data(
            KeyValues.AZURE_OPENAI_API_KEY
        )
        self.openai_api_version = KEY_FILE_HANDLER.fetch_data(
            KeyValues.OPENAI_API_VERSION
        )
        self.azure_embedding_deployment = KEY_FILE_HANDLER.fetch_data(
            KeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME
        )
        self.azure_endpoint = KEY_FILE_HANDLER.fetch_data(
            KeyValues.AZURE_OPENAI_ENDPOINT
        )
        self.args = args
        self.kwargs = kwargs
        super().__init__(self.azure_embedding_deployment)

    def load_model(self):
        return AzureOpenAIEmbeddings(
            openai_api_version=self.openai_api_version,
            azure_deployment=self.azure_embedding_deployment,
            azure_endpoint=self.azure_endpoint,
            openai_api_key=self.azure_openai_api_key,
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
