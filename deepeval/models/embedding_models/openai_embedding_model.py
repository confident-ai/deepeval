from typing import Dict, Optional, List
from openai import OpenAI, AsyncOpenAI
from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.models.retry_policy import (
    create_retry_decorator,
    sdk_retries_for,
)
from deepeval.constants import ProviderSlug as PS


retry_openai = create_retry_decorator(PS.OPENAI)

valid_openai_embedding_models = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]
default_openai_embedding_model = "text-embedding-3-small"


class OpenAIEmbeddingModel(DeepEvalBaseEmbeddingModel):

    def __init__(
        self,
        model: Optional[str] = None,
        client_kwargs: Optional[Dict] = None,
        **generation_kwargs,
    ):
        """
        Initializes an OpenAI embedding model.
        Required 'client_kwargs' values (if no env):
            - api_key

        Required env values (if no client_kwargs):
            - OPENAI_API_KEY

        You can pass in the **generation_kwargs for any generation settings you'd like to change
        """
        self.model_name = model if model else default_openai_embedding_model
        if self.model_name not in valid_openai_embedding_models:
            raise ValueError(
                f"Invalid model. Available OpenAI Embedding models: {', '.join(valid_openai_embedding_models)}"
            )
        self.client_kwargs = client_kwargs or {}
        self.generation_kwargs = generation_kwargs or {}

    @retry_openai
    def embed_text(self, text: str) -> List[float]:
        client = self.load_model(async_mode=False)
        response = client.embeddings.create(
            input=text, model=self.model_name, **self.generation_kwargs
        )
        return response.data[0].embedding

    @retry_openai
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model(async_mode=False)
        response = client.embeddings.create(
            input=texts, model=self.model_name, **self.generation_kwargs
        )
        return [item.embedding for item in response.data]

    @retry_openai
    async def a_embed_text(self, text: str) -> List[float]:
        client = self.load_model(async_mode=True)
        response = await client.embeddings.create(
            input=text, model=self.model_name, **self.generation_kwargs
        )
        return response.data[0].embedding

    @retry_openai
    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model(async_mode=True)
        response = await client.embeddings.create(
            input=texts, model=self.model_name, **self.generation_kwargs
        )
        return [item.embedding for item in response.data]

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
        if not sdk_retries_for(PS.OPENAI):
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
