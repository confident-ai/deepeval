from typing import Dict, List, Optional
from openai import AzureOpenAI, AsyncAzureOpenAI
from pydantic import SecretStr

from deepeval.config.settings import get_settings
from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.constants import ProviderSlug as PS
from deepeval.models.utils import require_secret_api_key


retry_azure = create_retry_decorator(PS.AZURE)


class AzureOpenAIEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        model: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        **client_kwargs,
    ):
        settings = get_settings()

        if openai_api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.openai_api_key: SecretStr | None = SecretStr(openai_api_key)
        else:
            self.openai_api_key = settings.AZURE_OPENAI_API_KEY

        self.openai_api_version = (
            openai_api_version or settings.OPENAI_API_VERSION
        )
        self.azure_endpoint = (
            azure_endpoint
            or settings.AZURE_OPENAI_ENDPOINT
            and str(settings.AZURE_OPENAI_ENDPOINT)
        )

        self.azure_deployment = (
            azure_deployment or settings.AZURE_EMBEDDING_DEPLOYMENT_NAME
        )
        self.client_kwargs = client_kwargs or {}
        self.model_name = model or self.azure_deployment
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(self.model_name)

    @retry_azure
    def embed_text(self, text: str) -> List[float]:
        client = self.load_model(async_mode=False)
        response = client.embeddings.create(
            input=text, model=self.model_name, **self.generation_kwargs
        )
        return response.data[0].embedding

    @retry_azure
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model(async_mode=False)
        response = client.embeddings.create(
            input=texts, model=self.model_name, **self.generation_kwargs
        )
        return [item.embedding for item in response.data]

    @retry_azure
    async def a_embed_text(self, text: str) -> List[float]:
        client = self.load_model(async_mode=True)
        response = await client.embeddings.create(
            input=text, model=self.model_name, **self.generation_kwargs
        )
        return response.data[0].embedding

    @retry_azure
    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model(async_mode=True)
        response = await client.embeddings.create(
            input=texts, model=self.model_name, **self.generation_kwargs
        )
        return [item.embedding for item in response.data]

    def get_model_name(self) -> str:
        return self.model_name

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(AzureOpenAI)
        return self._build_client(AsyncAzureOpenAI)

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.openai_api_key,
            provider_label="AzureOpenAI",
            env_var_name="AZURE_OPENAI_API_KEY",
            param_hint="`openai_api_key` to AzureOpenAIEmbeddingModel(...)",
        )

        client_kwargs = self.client_kwargs.copy()
        if not sdk_retries_for(PS.AZURE):
            client_kwargs["max_retries"] = 0

        client_init_kwargs = dict(
            api_key=api_key,
            api_version=self.openai_api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            **client_kwargs,
        )
        try:
            return cls(**client_init_kwargs)
        except TypeError as e:
            # older OpenAI SDKs may not accept max_retries, in that case remove and retry once
            if "max_retries" in str(e):
                client_init_kwargs.pop("max_retries", None)
                return cls(**client_init_kwargs)
            raise
