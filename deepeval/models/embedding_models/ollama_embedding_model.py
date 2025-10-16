from ollama import Client, AsyncClient
from typing import List, Optional, Dict

from deepeval.key_handler import EmbeddingKeyValues, KEY_FILE_HANDLER
from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.models.retry_policy import (
    create_retry_decorator,
)
from deepeval.constants import ProviderSlug as PS


retry_ollama = create_retry_decorator(PS.OLLAMA)


class OllamaEmbeddingModel(DeepEvalBaseEmbeddingModel):

    REQUIRED_KEY_MAPPING = {
        "host": EmbeddingKeyValues.LOCAL_EMBEDDING_BASE_URL,
    }

    def __init__(
        self,
        model: Optional[str] = None,
        client_kwargs: Optional[Dict] = None,
        **generation_kwargs,
    ):
        """
        Initializes an Ollama embedding model.
        Required 'client_kwargs' values (if no env):
            - host

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

    @retry_ollama
    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        response = embedding_model.embed(
            model=self.model_name, input=text, **self.generation_kwargs
        )
        return response["embeddings"][0]

    @retry_ollama
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        response = embedding_model.embed(
            model=self.model_name, input=texts, **self.generation_kwargs
        )
        return response["embeddings"]

    @retry_ollama
    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model(async_mode=True)
        response = await embedding_model.embed(
            model=self.model_name, input=text, **self.generation_kwargs
        )
        return response["embeddings"][0]

    @retry_ollama
    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model(async_mode=True)
        response = await embedding_model.embed(
            model=self.model_name, input=texts, **self.generation_kwargs
        )
        return response["embeddings"]

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(Client)
        return self._build_client(AsyncClient)

    def _build_client(self, cls):
        return cls(**self.client_kwargs)

    def get_model_name(self):
        return f"{self.model_name} (Ollama)"
