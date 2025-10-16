from openai import OpenAI, AsyncOpenAI
from typing import Dict, List, Optional

from deepeval.key_handler import EmbeddingKeyValues, KEY_FILE_HANDLER
from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.constants import ProviderSlug as PS


# consistent retry rules
retry_local = create_retry_decorator(PS.LOCAL)


class LocalEmbeddingModel(DeepEvalBaseEmbeddingModel):

    REQUIRED_KEY_MAPPING = {
        "api_key": EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY,
        "base_url": EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL,
    }

    def __init__(
        self,
        model: Optional[str] = None,
        client_kwargs: Optional[Dict] = None,
        **generation_kwargs,
    ):
        """
        Initializes a local embedding model.
        Required 'client_kwargs' values (if no env):
            - api_key
            - base_url

        Required env values (if no client_kwargs):
            - LOCAL_EMBEDDING_API_KEY
            - LOCAL_EMBEDDING_BASE_URL

        You can pass in the **generation_kwargs for any generation settings you'd like to change
        """
        self.client_kwargs = self._load_client_kwargs(client_kwargs) or {}
        self.model_name = model or KEY_FILE_HANDLER.fetch_data(
            EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME
        )
        self.generation_kwargs = generation_kwargs or {}
        if not self.model_name:
            raise ValueError(
                "Missing 'model'. Please pass it explicitly or set LOCAL_EMBEDDING_MODEL_NAME."
            )
        super().__init__(self.model_name)

    def _load_client_kwargs(self, client_kwargs: Optional[Dict]) -> Dict:
        if client_kwargs is not None:
            missing = [
                key
                for key in self.REQUIRED_KEY_MAPPING
                if key not in client_kwargs
            ]
            if missing:
                raise ValueError(
                    f"Missing required params in 'client_kwargs': {missing}"
                )
            return client_kwargs
        else:
            return {
                key: KEY_FILE_HANDLER.fetch_data(env_key)
                for key, env_key in self.REQUIRED_KEY_MAPPING.items()
            }

    @retry_local
    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        response = embedding_model.embeddings.create(
            model=self.model_name, input=[text], **self.generation_kwargs
        )
        return response.data[0].embedding

    @retry_local
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        response = embedding_model.embeddings.create(
            model=self.model_name, input=texts, **self.generation_kwargs
        )
        return [data.embedding for data in response.data]

    @retry_local
    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model(async_mode=True)
        response = await embedding_model.embeddings.create(
            model=self.model_name, input=[text], **self.generation_kwargs
        )
        return response.data[0].embedding

    @retry_local
    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model(async_mode=True)
        response = await embedding_model.embeddings.create(
            model=self.model_name, input=texts, **self.generation_kwargs
        )
        return [data.embedding for data in response.data]

    ###############################################
    # Model
    ###############################################

    def get_model_name(self):
        return self.model_name

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(OpenAI)
        return self._build_client(AsyncOpenAI)

    def _build_client(self, cls):
        client_kwargs = self.client_kwargs.copy()
        if not sdk_retries_for(PS.LOCAL):
            client_kwargs["max_retries"] = 0

        client_init_kwargs = {
            **client_kwargs,
        }
        try:
            return cls(**client_init_kwargs)
        except TypeError as e:
            # older OpenAI SDKs may not accept max_retries, in that case remove and retry once
            if "max_retries" in str(e):
                client_init_kwargs.pop("max_retries", None)
                return cls(**client_init_kwargs)
            raise
