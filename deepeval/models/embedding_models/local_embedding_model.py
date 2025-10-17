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
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        **client_kwargs,
    ):
        self.api_key = api_key or KEY_FILE_HANDLER.fetch_data(
            EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY
        )
        self.base_url = base_url or KEY_FILE_HANDLER.fetch_data(
            EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL
        )
        self.model_name = model or KEY_FILE_HANDLER.fetch_data(
            EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME
        )
        self.client_kwargs = client_kwargs or {}
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(self.model_name)

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

        client_init_kwargs = dict(
            api_key=self.api_key,
            base_url=self.base_url,
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
