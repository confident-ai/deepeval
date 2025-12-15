from typing import Dict, Optional, List
from openai import OpenAI, AsyncOpenAI
from pydantic import SecretStr

from deepeval.errors import DeepEvalError
from deepeval.config.settings import get_settings
from deepeval.models.utils import (
    require_secret_api_key,
    normalize_kwargs_and_extract_aliases,
)
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

_ALIAS_MAP = {
    "api_key": ["openai_api_key"],
}


class OpenAIEmbeddingModel(DeepEvalBaseEmbeddingModel):

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        normalized_kwargs, alias_values = normalize_kwargs_and_extract_aliases(
            "OpenAIEmbeddingModel",
            kwargs,
            _ALIAS_MAP,
        )

        # re-map depricated keywords to re-named positional args
        if api_key is None and "api_key" in alias_values:
            api_key = alias_values["api_key"]

        if api_key is not None:
            # keep it secret, keep it safe from serializings, logging and alike
            self.api_key: Optional[SecretStr] = SecretStr(api_key)
        else:
            self.api_key = get_settings().OPENAI_API_KEY

        model = model if model else default_openai_embedding_model
        if model not in valid_openai_embedding_models:
            raise DeepEvalError(
                f"Invalid model. Available OpenAI Embedding models: {', '.join(valid_openai_embedding_models)}"
            )
        self.kwargs = normalized_kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model)

    @retry_openai
    def embed_text(self, text: str) -> List[float]:
        client = self.load_model(async_mode=False)
        response = client.embeddings.create(
            input=text, model=self.name, **self.generation_kwargs
        )
        return response.data[0].embedding

    @retry_openai
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model(async_mode=False)
        response = client.embeddings.create(
            input=texts, model=self.name, **self.generation_kwargs
        )
        return [item.embedding for item in response.data]

    @retry_openai
    async def a_embed_text(self, text: str) -> List[float]:
        client = self.load_model(async_mode=True)
        response = await client.embeddings.create(
            input=text, model=self.name, **self.generation_kwargs
        )
        return response.data[0].embedding

    @retry_openai
    async def a_embed_texts(self, texts: List[str]) -> List[List[float]]:
        client = self.load_model(async_mode=True)
        response = await client.embeddings.create(
            input=texts, model=self.name, **self.generation_kwargs
        )
        return [item.embedding for item in response.data]

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(OpenAI)
        return self._build_client(AsyncOpenAI)

    def _build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="OpenAI",
            env_var_name="OPENAI_API_KEY",
            param_hint="`api_key` to OpenAIEmbeddingModel(...)",
        )

        client_kwargs = self.kwargs.copy()
        if not sdk_retries_for(PS.OPENAI):
            client_kwargs["max_retries"] = 0

        client_init_kwargs = dict(
            api_key=api_key,
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
        return f"{self.name} (OpenAI)"
