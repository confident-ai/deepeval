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
from deepeval.models.utils import (
    require_secret_api_key,
    normalize_kwargs_and_extract_aliases,
)
from deepeval.utils import require_param


retry_azure = create_retry_decorator(PS.AZURE)

_ALIAS_MAP = {
    "api_key": ["openai_api_key"],
    "base_url": ["azure_endpoint"],
    "deployment_name": ["azure_deployment"],
}


class AzureOpenAIEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        deployment_name: Optional[str] = None,
        api_version: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        normalized_kwargs, alias_values = normalize_kwargs_and_extract_aliases(
            "AzureOpenAIEmbeddingModel",
            kwargs,
            _ALIAS_MAP,
        )

        # re-map depricated keywords to re-named positional args
        if api_key is None and "api_key" in alias_values:
            api_key = alias_values["api_key"]
        if base_url is None and "base_url" in alias_values:
            base_url = alias_values["base_url"]
        if deployment_name is None and "deployment_name" in alias_values:
            deployment_name = alias_values["deployment_name"]

        settings = get_settings()

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = settings.AZURE_OPENAI_API_KEY

        api_version = api_version or settings.OPENAI_API_VERSION
        if base_url is not None:
            base_url = str(base_url).rstrip("/")
        elif settings.AZURE_OPENAI_ENDPOINT is not None:
            base_url = str(settings.AZURE_OPENAI_ENDPOINT).rstrip("/")

        deployment_name = (
            deployment_name or settings.AZURE_EMBEDDING_DEPLOYMENT_NAME
        )

        model = model or settings.AZURE_EMBEDDING_MODEL_NAME or deployment_name

        # validation
        self.deployment_name = require_param(
            deployment_name,
            provider_label="AzureOpenAIEmbeddingModel",
            env_var_name="AZURE_EMBEDDING_DEPLOYMENT_NAME",
            param_hint="deployment_name",
        )

        self.base_url = require_param(
            base_url,
            provider_label="AzureOpenAIEmbeddingModel",
            env_var_name="AZURE_OPENAI_ENDPOINT",
            param_hint="base_url",
        )

        self.api_version = require_param(
            api_version,
            provider_label="AzureOpenAIEmbeddingModel",
            env_var_name="OPENAI_API_VERSION",
            param_hint="api_version",
        )

        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = normalized_kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model)

    @retry_azure
    def embed_text(self, text: str) -> List[float]:
        client = self.load_model(async_mode=False)
        response = client.embeddings.create(
            input=text, model=self.name, **self.generation_kwargs
        )
        return response.data[0].embedding

    @retry_azure
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model(async_mode=False)
        response = client.embeddings.create(
            input=texts, model=self.name, **self.generation_kwargs
        )
        return [item.embedding for item in response.data]

    @retry_azure
    async def a_embed_text(self, text: str) -> List[float]:
        client = self.load_model(async_mode=True)
        response = await client.embeddings.create(
            input=text, model=self.name, **self.generation_kwargs
        )
        return response.data[0].embedding

    @retry_azure
    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model(async_mode=True)
        response = await client.embeddings.create(
            input=texts, model=self.name, **self.generation_kwargs
        )
        return [item.embedding for item in response.data]

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(AzureOpenAI)
        return self._build_client(AsyncAzureOpenAI)

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="AzureOpenAI",
            env_var_name="AZURE_OPENAI_API_KEY",
            param_hint="`api_key` to AzureOpenAIEmbeddingModel(...)",
        )

        client_kwargs = self.kwargs.copy()
        if not sdk_retries_for(PS.AZURE):
            client_kwargs["max_retries"] = 0

        client_init_kwargs = dict(
            api_key=api_key,
            api_version=self.api_version,
            azure_endpoint=self.base_url,
            azure_deployment=self.deployment_name,
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

    def get_model_name(self):
        return f"{self.name} (Azure)"
