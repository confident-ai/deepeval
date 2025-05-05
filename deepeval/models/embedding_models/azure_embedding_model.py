from typing import List
from openai import AzureOpenAI, AsyncAzureOpenAI
from deepeval.key_handler import KeyValues, KEY_FILE_HANDLER
from deepeval.models import DeepEvalBaseEmbeddingModel


class AzureOpenAIEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self):
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
        self.model_name = self.azure_embedding_deployment

    def embed_text(self, text: str) -> List[float]:
        client = self.load_model(async_mode=False)
        response = client.embeddings.create(
            input=text,
            model=self.azure_embedding_deployment,
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model(async_mode=False)
        response = client.embeddings.create(
            input=texts,
            model=self.azure_embedding_deployment,
        )
        return [item.embedding for item in response.data]

    async def a_embed_text(self, text: str) -> List[float]:
        client = self.load_model(async_mode=True)
        response = await client.embeddings.create(
            input=text,
            model=self.azure_embedding_deployment,
        )
        return response.data[0].embedding

    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model(async_mode=True)
        response = await client.embeddings.create(
            input=texts,
            model=self.azure_embedding_deployment,
        )
        return [item.embedding for item in response.data]

    def get_model_name(self) -> str:
        return self.model_name

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return AzureOpenAI(
                api_key=self.azure_openai_api_key,
                api_version=self.openai_api_version,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_embedding_deployment,
            )
        return AsyncAzureOpenAI(
            api_key=self.azure_openai_api_key,
            api_version=self.openai_api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_embedding_deployment,
        )
