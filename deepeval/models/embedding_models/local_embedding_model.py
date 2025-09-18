from openai import OpenAI, AsyncOpenAI
from typing import Dict, List

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
    def __init__(self, **kwargs):
        self.base_url = KEY_FILE_HANDLER.fetch_data(
            EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL
        )
        model_name = KEY_FILE_HANDLER.fetch_data(
            EmbeddingKeyValues.LOCAL_EMBEDDING_MODEL_NAME
        )
        self.api_key = KEY_FILE_HANDLER.fetch_data(
            EmbeddingKeyValues.LOCAL_EMBEDDING_API_KEY
        )
        self.kwargs = kwargs
        super().__init__(model_name)

    @retry_local
    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        response = embedding_model.embeddings.create(
            model=self.model_name,
            input=[text],
        )
        return response.data[0].embedding

    @retry_local
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        response = embedding_model.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [data.embedding for data in response.data]

    @retry_local
    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model(async_mode=True)
        response = await embedding_model.embeddings.create(
            model=self.model_name,
            input=[text],
        )
        return response.data[0].embedding

    @retry_local
    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model(async_mode=True)
        response = await embedding_model.embeddings.create(
            model=self.model_name,
            input=texts,
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

    def _client_kwargs(self) -> Dict:
        """
        If Tenacity manages retries, turn off OpenAI SDK retries to avoid double retrying.
        If users opt into SDK retries via DEEPEVAL_SDK_RETRY_PROVIDERS=local, leave them enabled.
        """
        kwargs = dict(self.kwargs or {})
        if not sdk_retries_for(PS.LOCAL):
            kwargs["max_retries"] = 0
        return kwargs

    def _build_client(self, cls):
        kw = dict(
            api_key=self.api_key,
            base_url=self.base_url,
            **self._client_kwargs(),
        )
        try:
            return cls(**kw)
        except TypeError as e:
            # Older OpenAI SDKs may not accept max_retries; drop and retry once.
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise
