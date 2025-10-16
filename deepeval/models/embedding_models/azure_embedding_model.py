from typing import Dict, List, Optional
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

    REQUIRED_KEY_MAPPING = {
        "api_key": ModelKeyValues.AZURE_OPENAI_API_KEY,
        "api_version": ModelKeyValues.OPENAI_API_VERSION,
        "azure_endpoint": ModelKeyValues.AZURE_OPENAI_ENDPOINT,
        "azure_deployment": EmbeddingKeyValues.AZURE_EMBEDDING_DEPLOYMENT_NAME,
    }

    def __init__(
        self,
        config: Optional[Dict] = None,
        **generation_kwargs
    ):
        """
        Initializes an Azure OpenAI embedding model.
        Please pass 'api_key', 'api_version', 'azure_endpoint' and 'azure_deployment' in your config if you're not using env

        Required env values (if no config):
            - AZURE_OPENAI_API_KEY
            - OPENAI_API_VERSION
            - AZURE_OPENAI_ENDPOINT
            - AZURE_EMBEDDING_DEPLOYMENT_NAME
        """
        self.config = self._load_config(config)
        self.generation_kwargs = generation_kwargs or {}
        self.model_name = self.config["azure_deployment"]

    def _load_config(self, config: Optional[Dict]) -> Dict:
        if config is not None:
            missing = [key for key in self.REQUIRED_KEY_MAPPING if key not in config]
            if missing:
                raise ValueError(f"Missing required params in 'config': {missing}")
            return config
        else:
            return {
                key: KEY_FILE_HANDLER.fetch_data(env_key)
                for key, env_key in self.REQUIRED_KEY_MAPPING.items()
            }

    @retry_azure
    def embed_text(self, text: str) -> List[float]:
        client = self.load_model(async_mode=False)
        response = client.embeddings.create(
            input=text,
            model=self.model_name,
        )
        return response.data[0].embedding

    @retry_azure
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model(async_mode=False)
        response = client.embeddings.create(
            input=texts,
            model=self.model_name,
        )
        return [item.embedding for item in response.data]

    @retry_azure
    async def a_embed_text(self, text: str) -> List[float]:
        client = self.load_model(async_mode=True)
        response = await client.embeddings.create(
            input=text,
            model=self.model_name,
        )
        return response.data[0].embedding

    @retry_azure
    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model(async_mode=True)
        response = await client.embeddings.create(
            input=texts,
            model=self.model_name,
        )
        return [item.embedding for item in response.data]

    def get_model_name(self) -> str:
        return self.model_name

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(AzureOpenAI)
        return self._build_client(AsyncAzureOpenAI)

    def _build_client(self, cls):
        generation_kwargs = self.generation_kwargs.copy()
        if not sdk_retries_for(PS.AZURE):
            generation_kwargs["max_retries"] = 0

        kw = {
            **self.config,
            **generation_kwargs,
        }
        try:
            return cls(**kw)
        except TypeError as e:
            # older OpenAI SDKs may not accept max_retries, in that case remove and retry once
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise
