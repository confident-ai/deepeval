from typing import List, Optional, Dict

from deepeval.config.settings import get_settings
from deepeval.utils import require_dependency
from deepeval.models import DeepEvalBaseEmbeddingModel
from deepeval.models.utils import (
    normalize_kwargs_and_extract_aliases,
)
from deepeval.models.retry_policy import (
    create_retry_decorator,
)
from deepeval.constants import ProviderSlug as PS
from deepeval.utils import require_param

retry_ollama = create_retry_decorator(PS.OLLAMA)

_ALIAS_MAP = {"base_url": ["host"]}


class OllamaEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        normalized_kwargs, alias_values = normalize_kwargs_and_extract_aliases(
            "OllamaEmbeddingModel",
            kwargs,
            _ALIAS_MAP,
        )

        # re-map depricated keywords to re-named positional args
        if base_url is None and "base_url" in alias_values:
            base_url = alias_values["base_url"]

        settings = get_settings()

        if base_url is not None:
            self.base_url = str(base_url).rstrip("/")
        elif settings.LOCAL_EMBEDDING_BASE_URL is not None:
            self.base_url = str(settings.LOCAL_EMBEDDING_BASE_URL).rstrip("/")
        else:
            self.base_url = "http://localhost:11434"

        model = model or settings.LOCAL_EMBEDDING_MODEL_NAME

        # validation
        model = require_param(
            model,
            provider_label="OllamaEmbeddingModel",
            env_var_name="LOCAL_EMBEDDING_MODEL_NAME",
            param_hint="model",
        )

        # Keep sanitized kwargs for client call to strip legacy keys
        self.kwargs = normalized_kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model)

    @retry_ollama
    def embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model()
        response = embedding_model.embed(
            model=self.name, input=text, **self.generation_kwargs
        )
        return response["embeddings"][0]

    @retry_ollama
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model()
        response = embedding_model.embed(
            model=self.name, input=texts, **self.generation_kwargs
        )
        return response["embeddings"]

    @retry_ollama
    async def a_embed_text(self, text: str) -> List[float]:
        embedding_model = self.load_model(async_mode=True)
        response = await embedding_model.embed(
            model=self.name, input=text, **self.generation_kwargs
        )
        return response["embeddings"][0]

    @retry_ollama
    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        embedding_model = self.load_model(async_mode=True)
        response = await embedding_model.embed(
            model=self.name, input=texts, **self.generation_kwargs
        )
        return response["embeddings"]

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        ollama = require_dependency(
            "ollama",
            provider_label="OllamaEmbeddingModel",
            install_hint="Install it with `pip install ollama`.",
        )

        if not async_mode:
            return self._build_client(ollama.Client)
        return self._build_client(ollama.AsyncClient)

    def _build_client(self, cls):
        return cls(host=self.base_url, **self.kwargs)

    def get_model_name(self):
        return f"{self.name} (Ollama)"
