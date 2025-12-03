from ollama import Client, AsyncClient, ChatResponse
from typing import Optional, Tuple, Union, Dict
from pydantic import BaseModel

from deepeval.config.settings import get_settings
from deepeval.models.retry_policy import (
    create_retry_decorator,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.constants import ProviderSlug as PS


retry_ollama = create_retry_decorator(PS.OLLAMA)


class OllamaModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0,
        generation_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        settings = get_settings()
        model_name = model or settings.LOCAL_MODEL_NAME
        self.base_url = (
            base_url
            or (
                settings.LOCAL_MODEL_BASE_URL
                and str(settings.LOCAL_MODEL_BASE_URL)
            )
            or "http://localhost:11434"
        )
        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")
        self.temperature = temperature
        # Raw kwargs destined for the underlying Ollama client
        self.kwargs = kwargs
        self.generation_kwargs = generation_kwargs or {}
        super().__init__(model_name)

    ###############################################
    # Other generate functions
    ###############################################

    @retry_ollama
    def generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[Union[str, Dict], float]:
        chat_model = self.load_model()
        response: ChatResponse = chat_model.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema() if schema else None,
            options={
                **{"temperature": self.temperature},
                **self.generation_kwargs,
            },
        )
        return (
            (
                schema.model_validate_json(response.message.content)
                if schema
                else response.message.content
            ),
            0,
        )

    @retry_ollama
    async def a_generate(
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Tuple[str, float]:
        chat_model = self.load_model(async_mode=True)
        response: ChatResponse = await chat_model.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            format=schema.model_json_schema() if schema else None,
            options={
                **{"temperature": self.temperature},
                **self.generation_kwargs,
            },
        )
        return (
            (
                schema.model_validate_json(response.message.content)
                if schema
                else response.message.content
            ),
            0,
        )

    ###############################################
    # Model
    ###############################################

    def load_model(self, async_mode: bool = False):
        if not async_mode:
            return self._build_client(Client)
        return self._build_client(AsyncClient)

    def _client_kwargs(self) -> Dict:
        """Return kwargs forwarded to the underlying Ollama Client/AsyncClient."""
        return dict(self.kwargs or {})

    def _build_client(self, cls):
        kw = dict(
            host=self.base_url,
            **self._client_kwargs(),
        )
        return cls(**kw)

    def get_model_name(self):
        return f"{self.model_name} (Ollama)"
