from typing import Dict, List
from openai import AzureOpenAI, AsyncAzureOpenAI
from deepeval.key_handler import (
    EmbeddingKeyValues,
    ModelKeyValues,
    KEY_FILE_HANDLER,
)
from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.constants import ProviderSlug as PS


retry_azure = create_retry_decorator(PS.AZURE)


class AzureOpenAIEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self, **kwargs):
        self.azure_openai_api_key = KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.AZURE_OPENAI_API_KEY
        )
        self.openai_api_version = KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.OPENAI_API_VERSION
        )
        self.azure_embedding_deployment = KEY_FILE_HANDLER.fetch_data(
            EmbeddingKeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME
        )
        self.azure_endpoint = KEY_FILE_HANDLER.fetch_data(
            ModelKeyValues.AZURE_OPENAI_ENDPOINT
        )
        self.model_name = self.azure_embedding_deployment
        self.kwargs = kwargs

    @retry_azure
    def embed_text(self, text: str) -> List[float]:
        client = self.load_model(async_mode=False)
        response = client.embeddings.create(
            input=text,
            model=self.azure_embedding_deployment,
        )
        return response.data[0].embedding

    @retry_azure
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model(async_mode=False)
        response = client.embeddings.create(
            input=texts,
            model=self.azure_embedding_deployment,
        )
        return [item.embedding for item in response.data]

    @retry_azure
    async def a_embed_text(self, text: str) -> List[float]:
        client = self.load_model(async_mode=True)
        response = await client.embeddings.create(
            input=text,
            model=self.azure_embedding_deployment,
        )
        return response.data[0].embedding

    @retry_azure
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
            return self._build_client(AzureOpenAI)
        return self._build_client(AsyncAzureOpenAI)

    def _client_kwargs(self) -> Dict:
        """
        If Tenacity is managing retries, force OpenAI SDK retries off to avoid double retries.
        If the user opts into SDK retries for 'azure' via DEEPEVAL_SDK_RETRY_PROVIDERS,
        leave their retry settings as is.
        """
        kwargs = dict(self.kwargs or {})
        if not sdk_retries_for(PS.AZURE):
            kwargs["max_retries"] = 0
        return kwargs

    def _build_client(self, cls):
        kw = dict(
            api_key=self.azure_openai_api_key,
            api_version=self.openai_api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_embedding_deployment,
            **self._client_kwargs(),
        )
        try:
            return cls(**kw)
        except TypeError as e:
            # older OpenAI SDKs may not accept max_retries, in that case remove and retry once
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise
